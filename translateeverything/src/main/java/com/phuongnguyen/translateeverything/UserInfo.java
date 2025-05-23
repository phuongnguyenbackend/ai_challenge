package com.phuongnguyen.translateeverything;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

@Document(collection = "user_info")
public class UserInfo {
    @Id
    private String username;
    private String password;
    private String fullName;
    private String gmailUser;

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public String getFullName() {
        return fullName;
    }

    public void setFullName(String fullName) {
        this.fullName = fullName;
    }
    public String getGmailUser(){
        return gmailUser;
    }
    public void setGmailUser(String gmailUser){
        this.gmailUser = gmailUser;
    }
}
