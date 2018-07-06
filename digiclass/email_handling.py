import smtplib
from string import Template
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from email.mime.base import MIMEBase
import os

class Email_Handling:

    def __init__(self, address, password, smtp):
        self.MY_ADDRESS = address
        self.PASSWORD = password
        self.SMTP = smtp

    def get_contacts(self, filename):
        """
        Return two lists names, emails containing names and email addresses
        read from a file specified by filename.
        """

        names = []
        emails = []
        with open(filename, mode='r', encoding='utf-8') as contacts_file:
            for a_contact in contacts_file:
                names.append(a_contact.split()[0])
                emails.append(a_contact.split()[1])
        return names, emails

    def read_template(self, filename):
        """
        Returns a Template object comprising the contents of the
        file specified by filename.
        """

        with open(filename, 'r', encoding='utf-8') as template_file:
            template_file_content = template_file.read()
        return Template(template_file_content)

    def main_email(self, type):
        names, emails = self.get_contacts('email-templates/mycontacts.txt') # read contacts

        if(type == "attendance"):
            message_template = self.read_template('email-templates/attendance_message.txt')
            path = "reports-data/attendance-csv/"
            data = os.listdir(path)

        elif(type == "emotion"):
            message_template = self.read_template('email-templates/emotion_message.txt')
            path = "reports-data/emotion-csv/"
            data = os.listdir(path)

        # set up the SMTP server
        s = smtplib.SMTP(host=self.SMTP, port=587)
        s.starttls()
        s.login(self.MY_ADDRESS, self.PASSWORD)

        # For each contact, send the email:
        for name, email in zip(names, emails):
            msg = MIMEMultipart()       # create a message

            # add in the actual person name to the message template
            message = message_template.substitute(PERSON_NAME=name.title())

            # Prints out the message body for our sake
            print(message)

            # setup the parameters of the message
            msg['From']=self.MY_ADDRESS
            msg['To']=email
            msg['Subject']=type

            # add in the message body
            msg.attach(MIMEText(message, 'plain'))

            with open(path + data[0], 'rb') as fp:
                msg1 = MIMEBase('application', "octet-stream")
                msg1.set_payload(fp.read())

            encoders.encode_base64(msg1)
            msg1.add_header('Content-Disposition', 'attachment', filename=os.path.basename(data[0]))
            msg.attach(msg1)

            # send the message via the server set up earlier.
            s.send_message(msg)
            del msg

            # Terminate the SMTP session and close the connection
        s.quit()
