import os
import urllib.request

def main():
    url = 'http://credit.customs.gov.cn/ccppserver/verifyCode/creator'
    saveDir = r'D:\EricYANG\HKUST\21Fall\comp4471\project\dataset\gifs'
    num_file = 1000
    num_crawl = 0


    # check if legal saveDir, change directory
    try:
        if not os.path.isdir(saveDir):
            os.mkdir(saveDir)
    finally:
        os.chdir(saveDir)
        print("Store Dir:", saveDir)
    
    
    while(True):
        print("Crawling gif", num_crawl)
        filepath = os.path.join(str(num_crawl) + '.gif')
        urllib.request.urlretrieve(url, filepath)
        num_crawl += 1
        if (num_crawl >= num_file):
            break
    
if __name__ == '__main__':
    main()