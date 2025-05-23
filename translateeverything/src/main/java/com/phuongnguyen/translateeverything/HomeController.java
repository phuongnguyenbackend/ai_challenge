package com.phuongnguyen.translateeverything;

import jakarta.servlet.http.HttpSession;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class HomeController {
    @GetMapping("/api/getNameToHello")
    public String getUsername(HttpSession session, Model model){
        UserInfo currentUser = (UserInfo) session.getAttribute("currentUser");
        model.addAttribute("user", currentUser);
        return currentUser.getFullName();
    }
}
