"""One command: refresh data, build features, retrain saved model, print latest ranking."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
COMMANDS=[['collect_stocks.py'],['collect_fundamentals.py','--refresh'],['build_fundamental_features.py'],['build_features.py'],['train_model.py'],['predict.py']]

def main():
    for args in COMMANDS:
        print('\n'+'='*60+'\nRUNNING '+ ' '.join(args)+'\n'+'='*60,flush=True)
        subprocess.run([sys.executable,str(ROOT/args[0])]+args[1:],cwd=ROOT,check=True)
if __name__=='__main__': main()
