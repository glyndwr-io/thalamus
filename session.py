import redis.asyncio as redis
import scrypt
import secrets
import os
import json

from fastapi import HTTPException, WebSocket, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, constr

r = redis.Redis()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token.json")

class Session(BaseModel):
    token: str
    public_key: str

class APIKeypair(BaseModel):
    public_key: constr(min_length=46, max_length=46)
    secret_key: constr(min_length=46, max_length=46)

async def getSession(token: str = Depends(oauth2_scheme)):
    public_key = await r.get(f"bearer_{token}")
    
    if(public_key == None):
        raise HTTPException(status_code=401)

    return Session(token=token, public_key=public_key)

async def getSessionWS(websocket: WebSocket):
    authorization = websocket.headers.get('authorization')
    invalid = {
        "event": "hangup",
        "message": "Invalid Authorization"
    }

    # Hangup if no authorization header
    if(authorization == None):
        await websocket.accept()
        await websocket.send_json(invalid)
        await websocket.close()
        return

    # Hangup if Bearer not formatted properly
    split = authorization.split(' ')
    if(len(split) != 2 or split[0].lower() != 'bearer'):
        await websocket.accept()
        await websocket.send_json(invalid)
        await websocket.close()
        return

    # Hangup if Bearer is invalid
    public_key = await r.get(f"bearer_{split[1]}")
    if(public_key == None):
        await websocket.accept()
        await websocket.send_json(invalid)
        await websocket.close()
        return

    return Session(token=split[1], public_key=public_key)

async def authenticate(keys: APIKeypair):
    ttl = 3600
    
    # Get the Encrypted Secret for the Public Key
    # Raise 401 if no public key found
    encrypted = await r.hget('apikeys', keys.public_key)
    if(encrypted == None):
        raise HTTPException(status_code=401)

    # Check if the provided secret is verified
    # by the encrypted stored secret. If so 
    # generate and store token
    scrypt.decrypt(encrypted, keys.secret_key, maxtime=0.5, encoding=None)
    token = secrets.token_urlsafe(32)
    await r.set(f"bearer_{token}", keys.public_key, ttl)
    return { "token": token, "ttl": ttl }


async def generateKeypair():
    public_key = f"pk_{secrets.token_urlsafe(32)}"
    secret_key = f"sk_{secrets.token_urlsafe(32)}"
    encrypted = scrypt.encrypt(os.urandom(64), secret_key, maxtime=0.5)

    await r.hset('apikeys', public_key, encrypted)

    return { "public_key": public_key, "secret_key": secret_key }

async def generate(path):
    keypair = await generateKeypair()
    print(path)
    with open(path, 'w') as f:
        json.dump(keypair, f, indent=4)
    return

async def revoke(public_key):
    await r.hdel('apikeys', public_key)
    return

async def purge():
    await r.flushdb()
    return