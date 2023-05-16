#! /usr/bin/python3
import argparse
import asyncio

from session import generate, revoke, purge

async def main():
    parser = argparse.ArgumentParser(description='Manage API Keys for Thalamus Core')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g', '--generate', action='store', 
        help='Generate a new keypair and store in the provided output directory')
    group.add_argument('-r', '--revoke', action='store', 
        help='Revoke an API keypair by it\'s Public Key')
    group.add_argument('-p', '--purge', action='store_true', help='Purge all API keypairs')

    args = parser.parse_args()

    if(args.purge == True):
        await purge()
    elif(args.revoke != None):
        await revoke(args.revoke)
    elif(args.generate != None):
        await generate(args.generate)

if __name__ == "__main__":
    asyncio.run(main())