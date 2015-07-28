#!/usr/bin/env python
# -*- coding: utf-8 -*-

import smtplib
from email.mime.text import MIMEText
import os, sys
from subprocess import check_output

if len(sys.argv) == 1:
	content = check_output(['git','log','-2'])
else:
	content = check_output(['git','--git-dir',sys.argv[1]+'/.git','log','-2'])
msg = MIMEText(content,'plain','utf-8')

sender = 'git'
recipients = ['rrausch@physnet.uni-hamburg.de','mpeschke@physnet.uni-hamburg.de']

msg['Subject'] = 'git merge'
msg['From'] = sender

for recipient in recipients:
	msg['To'] = recipient
	s = smtplib.SMTP('localhost')
	s.sendmail(sender, [recipient], msg.as_string())
	s.quit()
