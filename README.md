# ensemble-nerd

This repository aims to show two multilingual ensemble methods that combine the responses of web services **NER** and **NED** in order to improve the quality of the predicted entities.
Both methods represent the information got by the extractor responses as real-valued vector (features engineering) and use **Deep Neural Networks** to produce the final output.

5 ensemble models have been built using the training set related to these gold standard:
* aida
* oke2015
* oke2016
* neel2015
* french subtitles corpus

## Web API

The easiest way to try and use the **ensemble nerd** is via a Web API.

### Version information
Version : 0.1.0

### URI scheme
Host : http://enerd.eurecom.fr/

BasePath : TO_INSERT

Schemes : HTTP

### Paths

#### POST /entities

##### Description
Extract, type and link entities from a document.

##### Request

The format in the HTTP header is respectively **text/plain** or **application/json**. In the second case, the input JSON has to be like this:

```
{
  "text":<PLAIN_TEXT_TO_BE_ANNOTATED>
}
```

###### Parameters

| param | default |  description |
|:-------------:|:-------------:|:-------------:|
| lang| auto | string containing ISO-639-2 language code |
| model_recognition| "oke2016" | model recognition model to be used |
| model_disambiguation| "oke2016" | model disambiguation model to be used |

None of these parameters is mandatory. If the language is not specified, it is automatically detected by the program.

###### Example
A CURL POST request example is:
```
curl -X POST "TO_INSERT/entities?lang=en" -H "Content-type: application/json" -d '{"text":"In Italy the rector is the head of the university and Rappresentante Legale (Legal representative) of the university. He or she is elected by an electoral body."}'
```
It is identical to:

```
curl -X POST "http://127.0.0.1:5000/entities?lang=en" -H "Content-type: text/plain" -d "In Italy the rector is the head of the university and Rappresentante Legale (Legal representative) of the university. He or she is elected by an electoral body."
```

##### Response

The response format is **application/json**. An response example is showed [here](https://raw.githubusercontent.com/D2KLab/ensemble-nerd/master/app/response_samples/response1.json?token=ARExSVhO2fCJz8qLHTJLm6FT1uxRzVwqks5aumj8wA%3D%3D).


## Train your own model

Follow the guide inside the [`app` folder](./app).

## Docker setup

Build

    docker build -t d2klab/ensemble-nerd .

Run

    docker run -d -p 8089:80  -v /Users/pasquale/git/ensemble-nerd/app/data:/data --name enerd d2klab/ensemble-nerd

<!-- --restart=unless-stopped -->

Stop

    docker stop enerd
    docker rm enerd ##remove from available containers
    docker rmi d2klab/ensemble-nerd ##remove from images

