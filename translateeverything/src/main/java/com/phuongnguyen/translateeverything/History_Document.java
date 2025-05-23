package com.phuongnguyen.translateeverything;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

@Document(collection = "history_document")
public class History_Document {

    @Id
    private String id;
    private String idUser;
    private String filename;
    private String contentType;
    private byte[] data;
    private String type;

    public History_Document() {
    }


    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }
    public String getIdUser(){
        return idUser;
    }

    public void setIdUser(String idUSer) {
        this.idUser = idUSer;
    }

    public String getFilename() {
        return filename;
    }

    public void setFilename(String filename) {
        this.filename = filename;
    }

    public String getContentType() {
        return contentType;
    }

    public void setContentType(String contentType) {
        this.contentType = contentType;
    }

    public byte[] getData() {
        return data;
    }

    public void setData(byte[] data) {
        this.data = data;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }
}