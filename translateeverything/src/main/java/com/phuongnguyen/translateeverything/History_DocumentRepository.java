package com.phuongnguyen.translateeverything;

import org.springframework.data.mongodb.repository.MongoRepository;

import java.util.List;

public interface History_DocumentRepository extends MongoRepository<History_Document, String> {
    List<History_Document> findAllByIdUser(String idUser);

}
