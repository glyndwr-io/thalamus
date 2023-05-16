from tensorflow import keras
from keras.callbacks import Callback
from classifier import TrainingConfig

import asyncio

class Messaging():
    def __init__(self):
        self.queues = {}
        self.loop = asyncio.get_event_loop()

    def hasQueue(self, namespace: str):
        return namespace in self.queues

    def getQueue(self, namespace: str):
        return self.queues.get(namespace)

    def createQueue(self, namespace: str):
        if(not self.hasQueue(namespace)):
            self.queues[namespace] = asyncio.Queue()
        
        return self.queues[namespace]

    def dispenseQueue(self, namespace: str):
        if(self.hasQueue(namespace)):
            self.queues.pop(namespace)

    def sendThreadsafe(self, namespace: str, message: str):
        asyncio.run_coroutine_threadsafe(
            self.sendMessage(namespace, message), 
            self.loop
        )

    async def sendMessage(self, namespace: str, message: str):
        if(self.hasQueue(namespace)):
            await self.queues[namespace].put(message)

    async def getMessage(self, namespace: str):
        if(self.hasQueue(namespace)):
            return await self.queues[namespace].get()

# Callbacks to provide to Keras. Sends messages via
# Async Queue which are then relayed via WebSockets.
# If supplied, data will be sent via POST requests
# to webhook endpoints.
class MessageCallbacks(Callback):
    def __init__(self, namespace: str, messaging: Messaging, config: TrainingConfig):
        super().__init__()
        self.messaging = messaging
        self.namespace = namespace
        self.config = config.dict(exclude_none=True)

    def on_train_begin(self, logs):
        self.messaging.sendThreadsafe(self.namespace, {"event": "trainingStart"})

    def on_epoch_begin(self, epoch, logs):
        self.messaging.sendThreadsafe(self.namespace, {
            "event": "epochStart",
            "data": {
                "epoch": epoch+1,
                "totalEpochs": self.config.get('epochs'),
                "progress": (epoch) / self.config.get('epochs')
            }
        })

    def on_epoch_end(self, epoch, logs):
        print(logs)
        self.messaging.sendThreadsafe(self.namespace, {
            "event": "epochEnd",
            "data": {
                "epoch": epoch+1,
                "totalEpochs": self.config.get('epochs'),
                "progress": (epoch+1) / self.config.get('epochs'),
                "trainLoss": logs.get('loss'),
                "trainAccuracy": logs.get('accuracy'),
                "validLoss": logs.get('val_loss'),
                "validAccuracy": logs.get('val_accuracy'),
            }
        })
        
    def on_train_end(self, logs):
        # Message of Nonetype will result in closing of the websocket
        self.messaging.sendThreadsafe(self.namespace, {"event": "trainingEnd"})
        self.messaging.sendThreadsafe(self.namespace, None)