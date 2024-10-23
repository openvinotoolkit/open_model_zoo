from accuracy_checker.main import main
import sys

if __name__ == '__main__':
    sys.argv.extend(['-c', 'C:\\Users\\Atikin\\Desktop\\Programming\\open_model_zoo\\models\\public\\gcn\\accuracy-check.yml'])
    main()