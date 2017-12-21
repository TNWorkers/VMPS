#!/usr/bin/env python
# -*- coding: utf-8 -*-

import smtplib
from email.mime.text import MIMEText
import os, sys
from subprocess import check_output

# hook-Name: post-receive
# Inhalt: 
##!/bin/sh
#/afs/physnet.uni-hamburg.de/groups/group-th1_po/VMPS++/DMRG/send_mail.py /afs/physnet.uni-hamburg.de/groups/group-th1_po/VMPS++/DMRG/
#exit 0

# ohne Argumente: 
# git log -2
if len(sys.argv) == 1:
	content = check_output(['git','log','-2','--decorate'])
# erstes Argument gibt git-Ordner an
# Der branch wird automatisch ermittelt als der letzte gepushte
elif len(sys.argv) == 2:
	branch = check_output(['git','log','--pretty=oneline','--abbrev-commit','--decorate','-1','--all']) # finde den branch-Namen
	branch = branch.split(' ') # splitte string zu array
	branch = branch[1] # branch-Name ist an dieser Stelle
	branch = branch.replace('(','') # Klammer löschen
	branch = branch.replace(')','') # Klammer löschen
	content = check_output(['git','--git-dir',sys.argv[1]+'/.git','log','-2','--decorate','--branches',branch])
msg = MIMEText(content,'plain','utf-8')

sender = 'git'
recipients = ['rrausch@physnet.uni-hamburg.de','mpeschke@physnet.uni-hamburg.de']

if len(sys.argv) >= 1:
	msg['Subject'] = 'git update '+branch
else:
	msg['Subject'] = 'git update'
msg['From'] = sender

for recipient in recipients:
	msg['To'] = recipient
	s = smtplib.SMTP('localhost')
	s.sendmail(sender, [recipient], msg.as_string())
	s.quit()
