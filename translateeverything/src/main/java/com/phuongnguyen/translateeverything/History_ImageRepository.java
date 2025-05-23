package com.phuongnguyen.translateeverything;

import org.springframework.data.mongodb.repository.MongoRepository;

import java.util.List;

public interface History_ImageRepository extends MongoRepository<History_Image, String> {
    List<History_Image> findAllByIdUser(String idUser);
}
