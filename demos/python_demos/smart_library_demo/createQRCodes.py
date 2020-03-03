import sys, os
import argparse
import json
sys.path.append("src/modules")
import QR_generator as qr

def create_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, dest = 'lib',  default='library.json')
    parser.add_argument('-o', type=str, dest = 'out',  default='qr-codes')
    return parser.parse_args()

def main():
    args = create_argparse()
    gen = qr.QRgenerator()
    if (args.lib != None and  os.path.isfile(args.lib) and args.out != None):
        try:
            os.mkdir(args.out)
        except OSError:
            print ("Creation of the directory %s failed" % args.out)
        else:
            print ("Successfully created the directory %s " % args.out)
            with open(args.lib, 'r', encoding='utf-8') as lib:
                data = json.load(lib)
            
            for book in data['books']:
                strData = (str(book['id'])+ ' ' + book['title'] + ' ' + 
                        book['author'] + ' ' + book['publisher'] + ' ' +
                        str(book['year']))
                qr = gen.makeQR(strData)
                print(strData)
                qr.save(args.out + '/' + str(book['id']) + '.png')
    else:
        print('File or directory not exists!')

if __name__ == '__main__':
    sys.exit(main() or 0)