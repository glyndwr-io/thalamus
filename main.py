import io
import os
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, Depends
from fastapi.responses import FileResponse
from classifier import ClassifierWrapper, TrainingConfig
from PIL import Image
from threading import Thread
from messaging import MessageCallbacks, Messaging
from session import Session, getSession, getSessionWS, APIKeypair, authenticate

dependencies=[Depends(getSession)]
messaging = Messaging()
app = FastAPI()

def initClassifier(GUID: UUID):
    basepath = f"./{GUID}"
    if(not os.path.exists(f"./{GUID}")):
        raise HTTPException(status_code=404, detail="Dataset Not Found")
    return ClassifierWrapper(basepath)

@app.post("/token.json")
async def getToken(keypair: APIKeypair):
    tokenResp = await authenticate(keypair)
    return tokenResp

@app.post("/namespace.json", dependencies=dependencies)
async def createNamespace():
    namespace = uuid4()
    while os.path.exists(f"./{namespace}"):
        namespace = uuid4()

    os.mkdir(f"./{namespace}")
    os.mkdir(f"./{namespace}/datasets")
    os.mkdir(f"./{namespace}/models")

    return { "namespace": namespace }


@app.post("/{GUID}/datasets/{dataset}.json", dependencies=dependencies)
async def createDataset(GUID: UUID, dataset: str):

    if(not os.path.exists(f"./{GUID}/datasets")):
        raise HTTPException(status_code=404, detail="Namespace does not exist")
    
    if(os.path.exists(f"./{GUID}/datasets/{dataset}")):
        raise HTTPException(status_code=400, detail="Dataset already exists")

    os.mkdir(f"./{GUID}/datasets/{dataset}")
    return { "dataset": dataset }

@app.put("/{GUID}/datasets/{dataset}/{classname}.json", dependencies=dependencies)
async def uploadClassdata(GUID: UUID, dataset: str, classname: str, files: list[UploadFile]):

    if(not os.path.exists(f"./{GUID}/datasets")):
        raise HTTPException(status_code=404, detail="Namespace does not exist")
    
    if(not os.path.exists(f"./{GUID}/datasets/{dataset}")):
        raise HTTPException(status_code=400, detail="Dataset does not exist")

    if(not os.path.exists(f"./{GUID}/datasets/{dataset}/{classname}")):
        raise HTTPException(status_code=400, detail="Classname does not exist")

    uploaded = []
    rejected = []
    for file in files:
        try:
            f = open(f"./{GUID}/datasets/{dataset}/{classname}/{file.filename}", "wb")
            for line in file.file:
                f.write(line)
            f.close()
            uploaded.append(file.filename)
        except:
            rejected.append(file.filename)

    return { "uploaded": uploaded, "rejected": rejected }

@app.get("/{GUID}/datasets/{dataset}.json", dependencies=dependencies)
async def datasetInfo(GUID: UUID, dataset: str):
    classifier = initClassifier(GUID)

    if not classifier.hasDataset(dataset):
        raise HTTPException(status_code=404, detail="Dataset not found")

    data = {}
    for classname in classifier.getClassNames(dataset):
        print(classname)
        data[classname] = classifier.getClassData(dataset, classname)
    
    return data

@app.get("/{GUID}/datasets/{dataset}/{classname}.json", dependencies=dependencies)
async def classInfo(GUID: UUID, dataset: str, classname: str):
    classifier = initClassifier(GUID)

    if not classifier.hasDataset(dataset):
        raise HTTPException(status_code=404, detail="Dataset Not Found")
    if not classifier.hasClass(dataset, classname):
        raise HTTPException(status_code=404, detail="Class Not Found")

    return classifier.getClassData(dataset, classname)

@app.get("/{GUID}/datasets/{dataset}/{classname}/{classdata}", dependencies=dependencies)
async def classImage(GUID: UUID, dataset: str, classname: str, classdata: str):
    classifier = initClassifier(GUID)
    
    if(classifier.hasClassData(dataset, classname, classdata)):
        return FileResponse(f"./datasets/{dataset}/{classname}/{classdata}")

    raise HTTPException(status_code=404, detail="Classdata Not Found") 

@app.post("/{GUID}/models/{model}/predict.json", dependencies=dependencies)
async def predict(GUID: UUID, model: str, file: UploadFile):
    classifier = initClassifier(GUID)
    
    if not classifier.load(model):
        raise HTTPException(status_code=404, detail="Model not found")

    data = await file.read()

    image = Image.open(io.BytesIO(data))
    return classifier.predict(image)

@app.post("/{GUID}/models/{model}/train.json", dependencies=dependencies)
async def train(GUID: UUID, model: str, config: TrainingConfig):
    classifier = initClassifier(GUID)
    
    if not classifier.hasDataset(model):
        raise HTTPException(status_code=404, detail="Dataset Not Found")

    namespace = f"{GUID}/{model}"
    messaging.createQueue(namespace)
    worker = Thread(
        target=classifier.createAndSaveBG,
        args=(model, config, MessageCallbacks(namespace, messaging, config))
    )
    worker.start()
    return { "status": "training", "wsUpdates": f"/{GUID}/models/{model}/updates"}

@app.websocket("/{GUID}/models/{model}/updates")
async def updates(GUID: UUID, model: str, websocket: WebSocket, session: Session = Depends(getSessionWS)):
    try:
        # Wait for a connection, and get the
        # corresponding queue
        await websocket.accept()
        isConnected = True
        
        # Make sure the queue exists before proceeding,
        # and hangup if it does not
        namespace = f"{GUID}/{model}"
        if(os.path.exists(f"./{GUID}") and messaging.hasQueue(namespace)):
            await websocket.send_json({ "event": "connected" })
        else:
            await websocket.send_json({
                "event": "hangup",
                "message": "Model not found"
            })
            await websocket.close()
            isConnected = False
            return
        
        # Wait for messages to accumulate in the queue
        while isConnected:
            update = await messaging.getMessage(namespace)

            if(update == None):
                await websocket.send_json({
                    "event": "hangup",
                    "message": "Model queue closed"
                })
                await websocket.close()
                messaging.dispenseQueue(namespace)
                isConnected = False
                return
            else:
                await websocket.send_json(update)

                

    # Error or connection closed unexpectedly
    except:
        return