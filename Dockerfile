# Un Dockerfile sert à définir étape par étape comment construire une image Docker, c'est-à-dire un environnement prêt à exécuter une application de manière 
# reproductible, isolée et indépendante de la machine sur laquelle elle tourne. 

# Ce Dockerfile configure un environnement Python léger, installe les dépendances du projet, copie les fichiers de l'application, puis lance une API FastAPI
# avec Uvicorn dans un conteneur sécurisé utilisant un utilisateur non-root.

# Utilise une image légère de Python 3.12
FROM python:3.12-slim

# Définit le répertoire de travail
WORKDIR /code

# Copie le fichier des dépendances et installe les packages
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Crée un utilisateur non-root
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Définit le répertoire de travail pour cet utilisateur
WORKDIR $HOME/app

# Copie les fichiers du projet
COPY --chown=user . $HOME/app

# Lance l'application FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]