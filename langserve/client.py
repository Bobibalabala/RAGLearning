"""
@Author: bo.yang-a2302@aqara.com
@Date: 

    
"""
from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://0.0.0.0:8000/chain/")
r = remote_chain.invoke({"language": "italian", "text": "hi"})
print(r)