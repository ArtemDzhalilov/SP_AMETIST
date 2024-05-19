import pandas as pd
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Integer
from pgvector.sqlalchemy import Vector
import uuid
from ml import vectorize
from sqlalchemy.sql import text
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    api_key = Column(String, unique=True)
    username = Column(String, unique=True)
    password = Column(String)
class VectorDocument(Base):
    __tablename__ = "documentvectors"

    id = Column(Integer, primary_key=True)
    user = Column(String, ForeignKey("users.username"))
    text = Column(String)
    vector = Column(Vector)
class Database:
    def __init__(self):
        self.engine = create_engine("postgresql://postgres:postgres@localhost:5438/")
        self.Session = sessionmaker(bind=self.engine)
    @staticmethod
    def generate_api_key():
        api_key = uuid.uuid4().hex
        return api_key

    def automigrate(self):
        Base.metadata.create_all(self.engine)
    def update_all_documents(self):
        new_documents_list = vectorize(pd.read_excel("ametist_data.xlsx").values)
        session = self.Session()
        session.query(VectorDocument).delete()
        for document in new_documents_list:
            session.add(VectorDocument(user=document[0], vector=Vector(document[1])))
        session.commit()
        session.close()
    def register(self, username, password):
        session = self.Session()
        user = User(username=username, password=password, api_key=self.generate_api_key())
        session.add(user)
        session.commit()
        session.close()
        return user
    def login(self, username, password):
        session = self.Session()
        user = session.query(User).filter(User.username == username, User.password == password).first()
        session.close()
        return user
    def get_nearest_documents(self, vector, n):
        session = self.Session()
        vector = Vector(vector)
        documents = session.query(VectorDocument).filter(VectorDocument.vector.cosine_distance(vector) < 0.5).order_by(VectorDocument.vector.cosine_distance(vector)).first().all().text
        session.close()
        return documents
    def get_api_key(self, username):
        session = self.Session()
        user = session.query(User).filter(User.username == username).first()
        session.close()
        return user.api_key
