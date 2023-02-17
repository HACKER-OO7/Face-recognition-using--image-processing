import time
import serial

def send_sms(recipient,message ):

    phone = serial.Serial("COM3",  9600, timeout=5)
    try:
        time.sleep(1)
        phone.write(b'ATZ\r')
        time.sleep(1)
        phone.write(b'AT+CMGF=1\r')
        time.sleep(1)
        phone.write(b'AT+CMGS="' + recipient.encode() + b'"\r')
        time.sleep(1)
        phone.write(message.encode() + b"\r")
        time.sleep(1)
        phone.write(bytes([26]))
        time.sleep(1)
    finally:
        phone.close()

send_sms("+919773161802" , "Kashyap is Present in the Class.")
send_sms("+917777939034" , "Arpit is Present in the Class.")
send_sms("+919081174172" , "Pratik is Present in the Class.")















# import serial
# import time

# class TextMessage:
#     def __init__(self, recipient="+919773161802", message="TextMessage.content not set."):
#         self.recipient = recipient
#         self.content = message

#     def setRecipient(self, number):
#         self.recipient = number

#     def setContent(self, message):
#         self.content = message

#     def connectPhone(self):
#         self.ser = serial.Serial('COM3', 9600, timeout=5, xonxoff = False, rtscts = False, bytesize = serial.EIGHTBITS, parity = serial.PARITY_NONE, stopbits = serial.STOPBITS_ONE)
#         time.sleep(1)

#     def sendMessage(self):
#         self.ser.write(b'ATZ\r')
#         time.sleep(1)
#         self.ser.write(b'AT+CMGF=1\r')
#         time.sleep(1)
#         self.ser.write(b'AT+CMGS="' + self.recipient.encode() + b'"\r')
#         time.sleep(1)
#         self.ser.write(self.content + "\r")
#         time.sleep(1)
#         self.ser.write(chr(26))
#         time.sleep(1)

#     def disconnectPhone(self):
#         self.ser.close()

# sms = TextMessage("+919773161802"," i sent this message from my computer")
# sms.connectPhone()
# sms.sendMessage()
# sms.disconnectPhone()
# print ("message sent successfully")