package com.phuongnguyen.translateeverything;

import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.*;

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
    @Autowired
    History_DocumentRepository historyDocumentRepository;
    @PostMapping("/api/document_history")
    public ResponseEntity<byte[]> uploadAndTranslateDoc(@RequestParam("original") MultipartFile file, @RequestParam("translated") MultipartFile blob, HttpSession session, Model model) throws IOException {
        UserInfo currentUser = (UserInfo) session.getAttribute("currentUser");
        model.addAttribute("user", currentUser);
        History_Document original = new History_Document();
        original.setIdUser(currentUser.getUsername());
        original.setFilename(file.getOriginalFilename());
        original.setContentType(file.getContentType());
        original.setData(file.getBytes());
        original.setType("original");
        historyDocumentRepository.save(original);

        History_Document translated = new History_Document();
        translated.setFilename("translated_" + file.getOriginalFilename());
        translated.setContentType("application/pdf");
        translated.setData(blob.getBytes());
        translated.setType("translated");
        translated.setIdUser(currentUser.getUsername());
        historyDocumentRepository.save(translated);
        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_PDF)
                .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=\"translated_" + file.getOriginalFilename() + "\"")
                .body(blob.getBytes());
    }

    @GetMapping("/api/view_document_history")
    @ResponseBody
    public List<Map<String, String>> getAllDocHistory(HttpSession session) {
        UserInfo currentUser = (UserInfo) session.getAttribute("currentUser");
        String username = currentUser.getUsername();
        List<History_Document> documents = historyDocumentRepository.findAllByIdUser(username);
        List<Map<String, String>> result = new ArrayList<>();
        List<History_Document> originals = new ArrayList<>();
        Map<String, History_Document> translatedMap = new HashMap<>();
        for (History_Document doc : documents) {
            if ("original".equals(doc.getType())) {
                originals.add(doc);
            } else if ("translated".equals(doc.getType())) {
                String originalName = doc.getFilename().replace("translated_", "");
                translatedMap.put(originalName, doc);
            }
        }
        for (History_Document original : originals) {
            Map<String, String> item = new HashMap<>();
            item.put("input", "/api/document/" + original.getId());
            item.put("inputName", original.getFilename());

            History_Document translated = translatedMap.get(original.getFilename());
            if (translated != null) {
                item.put("output", "/api/document/" + translated.getId());
                item.put("outputName", translated.getFilename());
            }

            result.add(item);
        }

        return result;
    }
    @GetMapping("/api/document/{id}")
    public ResponseEntity<byte[]> getDocumentById(@PathVariable String id) {
        Optional<History_Document> docOpt = historyDocumentRepository.findById(id);
        if (docOpt.isEmpty()) {
            return ResponseEntity.notFound().build();
        }

        History_Document doc = docOpt.get();
        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_PDF)
                .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=\"" + doc.getFilename() + "\"")
                .body(doc.getData());
    }
    @Autowired
    History_ImageRepository historyImageRepository;
    @PostMapping(value="/api/image_history", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<byte[]> uploadAndTranslateImage(@RequestParam("original") MultipartFile file, @RequestParam("translated") MultipartFile blob, HttpSession session, Model model) throws IOException {
        UserInfo currentUser = (UserInfo) session.getAttribute("currentUser");
        model.addAttribute("user", currentUser);
        History_Image original = new History_Image();
        original.setIdUser(currentUser.getUsername());
        original.setFilename(file.getOriginalFilename());
        original.setContentType(file.getContentType());
        original.setData(file.getBytes());
        original.setType("original");
        historyImageRepository.save(original);

        History_Image translated = new History_Image();
        translated.setFilename("translated_" + file.getOriginalFilename());
        translated.setContentType("image/png");
        translated.setData(blob.getBytes());
        translated.setType("translated");
        translated.setIdUser(currentUser.getUsername());
        historyImageRepository.save(translated);
        return ResponseEntity.ok()
                .contentType(MediaType.IMAGE_PNG)
                .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=\"translated_" + file.getOriginalFilename() + "\"")
                .body(blob.getBytes());
    }
    @GetMapping("/api/view_image_history")
    @ResponseBody
    public List<Map<String, String>> getAllImageHistory(HttpSession session) {
        UserInfo currentUser = (UserInfo) session.getAttribute("currentUser");
        String username = currentUser.getUsername();
        List<History_Image> images = historyImageRepository.findAllByIdUser(username);
        List<Map<String, String>> result = new ArrayList<>();
        List<History_Image> originals = new ArrayList<>();
        Map<String, History_Image> translatedMap = new HashMap<>();
        for (History_Image img : images) {
            if ("original".equals(img.getType())) {
                originals.add(img);
            } else if ("translated".equals(img.getType())) {
                String originalName = img.getFilename().replace("translated_", "");
                translatedMap.put(originalName, img);
            }
        }
        for (History_Image original : originals) {
            Map<String, String> item = new HashMap<>();
            item.put("input", "/api/image/" + original.getId());
            item.put("inputName", original.getFilename());

            History_Image translated = translatedMap.get(original.getFilename());
            if (translated != null) {
                item.put("output", "/api/image/" + translated.getId());
                item.put("outputName", translated.getFilename());
            }

            result.add(item);
        }

        return result;
    }
    @GetMapping("/api/image/{id}")
    public ResponseEntity<byte[]> getImageById(@PathVariable String id) {
        Optional<History_Image> imgOpt = historyImageRepository.findById(id);
        if (imgOpt.isEmpty()) {
            return ResponseEntity.notFound().build();
        }

        History_Image img = imgOpt.get();
        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_PDF)
                .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=\"" + img.getFilename() + "\"")
                .body(img.getData());
    }
}
