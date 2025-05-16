package com.phuongnguyen.translateeverything;

import jakarta.servlet.http.HttpSession;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
public class CheckSessionController {
    @GetMapping("/api/check_session_profile")
    public ResponseEntity<?> checkSessionProfile(HttpSession session) {
        Object user = session.getAttribute("currentUser");

        if (user != null) {
            return ResponseEntity.ok().body("Session valid");
        } else {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("Session invalid");
        }
    }
    @GetMapping("/api/check_session_home")
    public ResponseEntity<?> checkSessionHome(HttpSession session) {
        Object user = session.getAttribute("currentUser");

        if (user != null) {
            return ResponseEntity.ok().body("Session valid");
        } else {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("Session invalid");
        }
    }
    @GetMapping("/api/check_session_text_history")
    public ResponseEntity<?> checkSessionTextHistory(HttpSession session) {
        Object user = session.getAttribute("currentUser");

        if (user != null) {
            return ResponseEntity.ok().body("Session valid");
        } else {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("Session invalid");
        }
    }
    @GetMapping("/api/check_session_text_translate")
    public ResponseEntity<?> checkSessionTextTranslate(HttpSession session) {
        Object user = session.getAttribute("currentUser");

        if (user != null) {
            return ResponseEntity.ok().body("Session valid");
        } else {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("Session invalid");
        }
    }
}
