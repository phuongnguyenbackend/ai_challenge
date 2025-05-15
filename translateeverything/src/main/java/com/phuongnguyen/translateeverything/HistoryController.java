package com.phuongnguyen.translateeverything;

import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
public class HistoryController {
    @Autowired
    History_TextRepository historyTextRepository;
    @PostMapping("/api/text_history")
    public void saveTextTranslated(@RequestBody History_Text historyText, HttpSession session, Model model){
        UserInfo currentUser = (UserInfo) session.getAttribute("currentUser");
        model.addAttribute("user", currentUser);
        historyText.setIdUser(currentUser.getUsername());
        historyTextRepository.save(historyText);
    }
    @GetMapping("/api/view_text_history")
    public List<History_Text> getAllPosts(HttpSession session, Model model) {
        UserInfo currentUser = (UserInfo) session.getAttribute("currentUser");
        model.addAttribute("user", currentUser);
        return historyTextRepository.findAllByIdUser(currentUser.getUsername());
    }
}
