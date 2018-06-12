import pandas as pd
import numpy as np
from config import FILE_PATH

def main():
    df3 = pd.read_csv('output/sub-dsbowl_class3.csv')
    df0 = pd.read_csv('output/sub-dsbowl_class0.csv')

    sub = pd.concat([df3,df0])
    sub.to_csv(FILE_PATH+'sub-dsbowl_1'+'.csv', index=False)

if __name__=='__main__':
    main()
