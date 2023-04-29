This is our NLP final project. It is a python model that generates a movie synopsis using the movie script.

To run the program, first create and activate a virtual python environment. This can be done using the following commands in the repo directory:

Windows

```
python -m venv .venv
.venv/bin/activate
```

MacOS and Linux

```
python -m venv .venv
source .venv/bin/activate
```

After a virtual environment is created, install the dependencies using:

```
pip install -r requirements.txt
```

Once the environment is set up, you can run the program like so:

```
python eval.py
```

Preprocssing done on Movie Scripts:

- Movie Script DB use many line breaks and tags to format correctly on web viewer
  - removed all tags like <b>, <i>, etc
- Not too much consistency to how screenplays are layed out
  - Most likely since scripts are user submissions
