from __future__ import print_function
from asyncio.windows_events import NULL
import sys, logging

from gsmmodem.modem import GsmModem, SentSms
from gsmmodem.exceptions import TimeoutException

def parseArgs():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Simple script for sending SMS messages')
    parser.add_argument('-p', '--pin', metavar='PIN', default=None, help='SIM card PIN')
    parser.add_argument('-w', '--wait', type=int, default=0, help='Wait for modem to start, in seconds')
    return parser.parse_args()



    
def send_sms(args):
    BAUDRATE = 9600
    PORT = "COM3"
    destination = "+917777939034"
    text = "I'm Groot!"
    modem = GsmModem(PORT, BAUDRATE, AT_CNMI='')
    modem.connect(args.pin, waitingForModemToStartInSeconds=args.wait)
    print('Connecting to GSM modem on {0}...'.format(PORT))
    modem.connect()
    #modem.connect('0000',waitingForModemToStartInSeconds=0)
    print('Checking for network coverage...')
    try:
        print("Trying to reach coverage")
        modem.waitForNetworkCoverage(5)
    except TimeoutException:
        print('Network signal strength is not sufficient, please adjust modem position/antenna and try again.')
        modem.close()
        sys.exit(1)
            
    print('\nSending SMS message...')         
                
    try:
        sms = modem.sendSms(destination, text, waitForDeliveryReport='store_true')
    except TimeoutException:
        print('Failed to send message: the send operation timed out')
        modem.close()
        sys.exit(1)
    else:
        modem.close()
        if sms.report:
            print('Message sent{0}'.format(' and delivered OK.' if sms.status == SentSms.DELIVERED else ', but delivery failed.'))
        else:
            print('Message sent.')
                


args = parseArgs()
send_sms(args)

