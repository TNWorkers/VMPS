#!/usr/bin/env python
# -*- coding: utf-8 -*-

import smtplib
from email.mime.text import MIMEText
import os, sys
from subprocess import check_output

# ohne Argumente: 
#git log -2
if len(sys.argv) == 1:
	content = check_output(['git','log','-2','--decorate'])
# erstes Argument gibt git-Ordner an: 
#git --git-dir <argv[1]>/.git log -2
elif len(sys.argv) == 2:
	content = check_output(['git','--git-dir',sys.argv[1]+'/.git','log','-2','--decorate'])
# drittes Argument gibt den branch an:
# git --git-dir <argv[1]>/.git -branch <argv[2]> log -2
elif len(sys.argv) == 3:
	content = check_output(['git','--git-dir',sys.argv[1]+'/.git','log','-2','--decorate',sys.argv[2]])
msg = MIMEText(content,'plain','utf-8')

sender = 'git'
recipients = ['rrausch@physnet.uni-hamburg.de','mpeschke@physnet.uni-hamburg.de'

name = sys.argv[1].split('/')[-1]

if len(sys.argv) >= 1:
	msg['Subject'] = 'git update '+name
else:
	msg['Subject'] = 'git update'
msg['From'] = sender

for recipient in recipients:
	msg['To'] = recipient
	s = smtplib.SMTP('localhost')
	s.sendmail(sender, [recipient], msg.as_string())
	s.quit()
