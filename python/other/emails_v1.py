import smtplib

TO = 'rpatel93@gmail.com'
SUBJECT = 'Send email to rushi'
TEXT = 'body messege'

#credentials

gmail_sender = 'aura.capstone@gmail.com'
gmail_passwd = 'Capstone1'

server = smtplib.SMTP('smtp.gmail.com',587)
server.ehlo()
server.starttls()
server.ehlo()
server.login(gmail_sender,gmail_passwd)

BODY ='\r\n'.join([
	'To %s' % TO,
	'From %s' % gmail_sender,
	'Subject: %s' % SUBJECT,
	'',
	TEXT
	])

try:
	server.sendmail(gmail_sender,[TO],BODY)
	print 'email sent'
except:
	print 'error sending email'

server.quit()
