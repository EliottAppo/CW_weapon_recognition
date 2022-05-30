import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

#Le compte google utilisé pour envoyer le mail est : 
#login : reco.weapon.detection@gmail.com
#pass : reco6Weapon

def envoie(mail): #envoie un mail

    msg = MIMEMultipart()
    msg['From'] = 'reco.weapon.detection@gmail.com'
    msg['To'] ='eliottxbox@gmail.com'  #compte recevant le mail
    msg['Subject'] = 'ALERTE' 
    message = 'Arme détecté par le système'
    msg.attach(MIMEText(message))
    mailserver = smtplib.SMTP('smtp.gmail.com', 587)
    mailserver.ehlo()
    mailserver.starttls()
    mailserver.ehlo()
    mailserver.login('reco.weapon.detection@gmail.com', 'reco6Weapon')
    mailserver.sendmail('reco.weapon.detection@gmail.com', 'eliottxbox@gmail.com', msg.as_string())
    mailserver.quit()

