import boto3
def readText():
    results=[]
    imageSource=open('inputs/result.jpg','rb')
    client=boto3.client('textract')
    response=client.detect_document_text(Document={'Bytes':imageSource.read()})
    for item in response["Blocks"]:
        if item["BlockType"]=="LINE":
            lp=(item["Text"])
            #print(lp)
            results.append(lp)
    try:
        lens=[]
        print(results)
        for i in results:
            lens.append(len(i))
        return(results[lens.index(max(lens))])
    except:
        return(None)
