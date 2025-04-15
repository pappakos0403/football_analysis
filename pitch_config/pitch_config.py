from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class FootballPitchConfiguration: # Pálya konfiguráció [cm]
    width: int = 6800  # Pálya szélessége
    length: int = 10500  # Pálya hossza
    penalty_box_width: int = 4032  # Büntetőterület szélessége
    penalty_box_length: int = 1650  # Büntetőterület hossza
    goal_box_width: int = 1832  # Tizenhatos terület szélessége
    goal_box_length: int = 550  # Tizenhatos terület hossza
    centre_circle_radius: int = 915  # Középkör sugara
    penalty_spot_distance: int = 1100  # Büntetőpont távolsága a vonaltól

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        # Pálya pontjainak koordinátái [cm-ben]
        return [
            (0, 0),  # 1. pont: bal alsó sarok
            (0, (self.width - self.penalty_box_width) / 2),  # 2. pont: bal oldal, büntetőterület kezdete
            (0, (self.width - self.goal_box_width) / 2),  # 3. pont: bal oldal, tizenhatos terület kezdete
            (0, (self.width + self.goal_box_width) / 2),  # 4. pont: bal oldal, tizenhatos terület vége
            (0, (self.width + self.penalty_box_width) / 2),  # 5. pont: bal oldal, büntetőterület vége
            (0, self.width),  # 6. pont: bal felső sarok
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7. pont: tizenhatos terület belső pontja (közel a kapuhoz)
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8. pont: tizenhatos terület külső pontja (közel a kapuhoz)
            (self.penalty_spot_distance, self.width / 2),  # 9. pont: büntetőpont koordinátája (vízszintesen eltolva)
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10. pont: büntetőterület vonal belső pontja (közelebb a kapuhoz)
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11. pont: tizenhatos terület vonal belső részének kezdete
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12. pont: tizenhatos terület vonal belső részének vége
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13. pont: büntetőterület vonal külső pontja
            (self.length / 2, 0),  # 14. pont: középső alsó rész (pálya közepén, alul)
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15. pont: középkör alsó széle
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16. pont: középkör felső széle
            (self.length / 2, self.width),  # 17. pont: középső felső rész (pálya közepén, felül)
            (
                self.length - self.penalty_box_length,
                (self.width - self.penalty_box_width) / 2
            ),  # 18. pont: jobb oldali büntetőterület vonal belső pontja
            (
                self.length - self.penalty_box_length,
                (self.width - self.goal_box_width) / 2
            ),  # 19. pont: jobb oldali tizenhatos terület vonal belső részének kezdete
            (
                self.length - self.penalty_box_length,
                (self.width + self.goal_box_width) / 2
            ),  # 20. pont: jobb oldali tizenhatos terület vonal belső részének vége
            (
                self.length - self.penalty_box_length,
                (self.width + self.penalty_box_width) / 2
            ),  # 21. pont: jobb oldali büntetőterület vonal külső pontja
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22. pont: jobb oldali büntetőpont koordinátája
            (
                self.length - self.goal_box_length,
                (self.width - self.goal_box_width) / 2
            ),  # 23. pont: jobb oldali tizenhatos terület vonalának belső pontja
            (
                self.length - self.goal_box_length,
                (self.width + self.goal_box_width) / 2
            ),  # 24. pont: jobb oldali tizenhatos terület vonalának külső pontja
            (self.length, 0),  # 25. pont: jobb alsó sarok
            (self.length, (self.width - self.penalty_box_width) / 2),  # 26. pont: jobb oldal, büntetőterület belső vonala
            (self.length, (self.width - self.goal_box_width) / 2),  # 27. pont: jobb oldal, tizenhatos terület belső vonala
            (self.length, (self.width + self.goal_box_width) / 2),  # 28. pont: jobb oldal, tizenhatos terület külső vonala
            (self.length, (self.width + self.penalty_box_width) / 2),  # 29. pont: jobb oldal, büntetőterület külső vonala
            (self.length, self.width),  # 30. pont: jobb felső sarok
            (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31. pont: középkör bal szélének koordinátája
            (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32. pont: középkör jobb szélének koordinátája
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
        (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
    ])  # Pontok összekötése

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
        "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
        "14", "19"
    ])  # Pontok azonosítói

    colors: List[str] = field(default_factory=lambda: [
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#FF6347",
        "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#00BFFF", "#00BFFF"
    ])  # Pontok színei
