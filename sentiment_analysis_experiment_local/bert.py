from bert_serving.client import BertClient
bc = BertClient(ip='localhost', port=5555)
print(bc.encode(['First do it', 'then do it right', 'then do it better']))
