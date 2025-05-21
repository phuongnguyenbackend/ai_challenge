package com.phuongnguyen.translateeverything;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.Objects;
import java.util.Optional;

@RestController
public class UserController {
    @Autowired
    UserRepository userRepository;
    @PostMapping("/api/register")
    public String saveUserInfo(@RequestBody UserInfo userInfo){
        Optional<UserInfo> optionalUserInfo = userRepository.findById(userInfo.getUsername());
        if(optionalUserInfo.isPresent()){
            return "This account already exists";
        }else{
            userRepository.save(userInfo);
            return "a";
        }
    }
    @GetMapping("/api/login")
    public String CheckUserInfo(@RequestParam String username, @RequestParam String password,
                                HttpSession session){
        Optional <UserInfo> optionalUserInfo = userRepository.findById(username);
        if (optionalUserInfo.isPresent()){
            UserInfo temp = optionalUserInfo.get();
            if (Objects.equals(temp.getPassword(), password)){
                session.setAttribute("currentUser", temp);
                return "YES";
            }else{
                return "Wrong password";
            }
        }else{
            return "This account does not exist.";
        }
    }
    @GetMapping("/api/getInfoUser")
    public UserInfo getInfoUser(HttpSession session, Model model){
        UserInfo currentUser = (UserInfo) session.getAttribute("currentUser");
        model.addAttribute("user", currentUser);
        return currentUser;
    }
    @PostMapping("/api/updateInfoUser")
    public String UpdateUserInfo(@RequestBody UserUpdate userUpdate){
        Optional<UserInfo> temp = userRepository.findById(userUpdate.getUsername());
        UserInfo userInfo1 = temp.get();
        UserInfo userInfo = new UserInfo();
        if (userInfo1.getPassword().equals(userUpdate.getPassword())){
            if (userUpdate.getNewPassword().equals(userUpdate.getConfirmNewPassword())){
                userInfo.setUsername(userUpdate.getUsername());
                userInfo.setPassword(userUpdate.getConfirmNewPassword());
                userInfo.setFullName(userUpdate.getFullName());
                userInfo.setGmailUser(userUpdate.getEmailUSer());
                userRepository.save(userInfo);
                return "YES";
            }else{
                return "new password not equal to confirm password";
            }
        }else{
            return "Wrong password";
        }
    }
    @PostMapping("/api/logout")
    public ResponseEntity<String> logout(HttpServletRequest request) {
        HttpSession session = request.getSession(false);
        if (session != null) {
            session.invalidate();
        }
        return ResponseEntity.ok("Logged out");
    }

}
