import os
import urllib.request

def main():
    url = 'http://credit.customs.gov.cn/ccppserver/verifyCode/creator'
    saveDir = r'dataset\gifs'
    filenum = 2000


    # check if legal saveDir, change directory
    try:
        if not os.path.isdir(saveDir):
            os.mkdir(saveDir)
    finally:
        os.chdir(saveDir)
        print("Store Dir:", saveDir)
    

    while(True):
        filenum += 1
        print("Crawling gif", filenum)
        filepath = os.path.join(str(filenum) + '.gif')
        urllib.request.urlretrieve(url, filepath)
        
        if (filenum >= 3000):
            break
    
if __name__ == '__main__':
    main()