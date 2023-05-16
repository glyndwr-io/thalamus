# Thalamus - Machine Learning Microservice

Thalamus is a machine learning microservice written in Python using Reids, FastAPI and TensorFlow. It is designed to be easily scalable (both vertically and horizontally), containerizable and completely headless.

## Initialization

First, make sure you have all the depenencies required. The project requires Python 3 and Redis. If you do not have Redis installed on your system but do have Docker, you can easily spin up a server for testing as follows:

``docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest``

From there, make sure you have the following Python dependencies installed:

* FastAPI
* TensorFlow
* PIL
* Redis
* Scrypt

Once dependencies are ready, you can generate an API keypair. The keys manager is the only way to generate keys for security purposes. To generate a key, run:

```
./keys.py -g <outputfile>
```

The Public and Secret keys will then be stored in the specified file. This way the keys don't end up in your shell's history. If a key does get accidentally exposed you can revoke it like so:

```
./keys.py -r <public_key>
```

If someone has already used the compromised key to generate a Bearer Token, you can purge all keys and tokens like so:

```
./keys.py -p
```

With your keys generated, you can now start the server:

```
./start.sh
```

## Access

To access any endpoint, you must first authorize your keys with the server to recieve a Bearer token. This is then provided in the Authorization header on all subsiquent requests. These tokens expire, and the token's TTL is provided in the token response. To get a token, make the following request:

```
POST http://localhost:8000/token.json
content-type: application/json

{
    "public_key": "<Your Public Key>",
    "secret_key": "<Your Private Key>"
}
```

And you will recieve the following response:

```
{
    "token": "<Your Token Here>",
    "ttl": <Token TTL in Seconds>
}
```

Which you provide as an Authorization header in all of your requests:

```
Authorization: Bearer <Your Token Here>
```

## Documentation

Once the server is running, you can view the docs at:

```
http://localhost:8000/docs
```
