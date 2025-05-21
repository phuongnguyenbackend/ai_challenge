package com.phuongnguyen.translateeverything;

import org.springframework.data.mongodb.repository.MongoRepository;

import java.util.List;

public interface History_TextRepository extends MongoRepository<History_Text, String> {
    List<History_Text> findAllByIdUser(String idUser);
}
