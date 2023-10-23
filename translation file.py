import pandas as pd
import csv
import re
import os
import json
from googletrans import Translator

files = [f for f in os.listdir('.') if os.path.isfile(f)]
for filename in files:
    resultFile=filename.replace(".csv","")
    if filename.endswith(".csv") and not os.path.exists(resultFile+".json"):     
        print("Processing the "+filename)
        with open(filename) as f:
            with open("2"+filename, "w") as f1:
                for line in f:
                    lines=re.findall(r'\{.*?\}', line)
                    if (len(lines)>0):
                        line2=lines[0]+"\n"
                        line3=line2.replace('  ',' ').replace(' ','').replace('""','"').replace('""','"').replace('":','":"').replace('":""','":"').replace('}','"}').replace('""}','"}')
                        line4=line3.replace('{', '').replace('}', '')
                        f1.write(line4)
        f.close()
        f1.close()



        count=0
        translator = Translator()
        with open("2"+filename) as f:
            with open("3"+ filename,"w") as f1:
                for line in f:
                    lines=re.split('"";""',line.replace('";"','"";""'))
                    if (count==0):
                        header=""
                        for element in lines:
                            elements=re.split('"":""',element.replace('":"','"":""'))
                            header=header+'"";""'+elements[0]
                            count=count+1
                        f1.write((header.replace('""','"').replace('""','"')+'"')[2:])
                        f1.write("\n")
                    row="\n"#"\n"+'"'
                    for element in lines:
                        #print(element)
                        #elements=re.split('":"',element.replace('"','').replace(':','":"'))
                        elements = re.split('"":""', element.replace('":"', '"":""'))
                        #print(elements)
                        #elem=elements.replace('""','"').replace('""','"')
                        if (len(elements)!=2):
                            print(element)
                            print(elements)
                            #print(elem)
                            row=(row+'";"'+(element.replace(elements[0],"").replace('""','"')).replace(';',''))
                            print(row)
                        else:
                            row=(row+'";"'+((elements[1]).replace('""','"')).replace(';',''))
                            #finalrow=row.replace('""','"').replace('""','"')
                        row = translator.translate(row, src='ch', dest='en')          
                    f1.write((row+'"')[3:])
        f.close()
        f1.close()


        #import CSV file with differing number of columns per row
        print("Read the "+"3"+ filename+" for processing")
        dfa = pd.read_csv("3"+ filename,sep=';',encoding='utf8', quoting=csv.QUOTE_NONE, on_bad_lines='skip')#,header =None,names=range(23))
        dfresult = dfa.dropna().replace('""','"', regex=True)
        dfresult.iloc[:,:23].to_csv("F"+filename,sep=";" , header=True)

        JSON=dfresult.to_json(orient="records")
        # Writing to sample.json
        with open(filename+".json", "w") as outfile:
            outfile.write(JSON)
        
        os.remove("2"+filename) 
        os.remove("3"+filename) 
        os.remove("F"+filename) 
        os.remove(filename+".json")         
        
        parsedLines = json.loads(JSON)
        with open("F"+filename,"w") as f1:
            for line in parsedLines:
                string = json.dumps(line)
                #print(string)
                f1.write(string+"\n")
        f1.close()                
                    
                    
        
        
        



