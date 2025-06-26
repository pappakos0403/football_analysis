from nicegui import ui, events, app
from pathlib import Path
from main_restructured import run_analysis_pipeline
import asyncio
from concurrent.futures import ThreadPoolExecutor
import shutil
import os
from collections import defaultdict, OrderedDict

# ----------------
# --- VÁLTOZÓK ---
# ----------------

# Feltöltött videók könyvtára
INPUT_DIR = Path("input_videos")
INPUT_DIR.mkdir(exist_ok=True)

# Elemzett videók könyvtára
OUTPUT_DIR = Path("output_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

# Logó könyvtára
Path("logo").mkdir(exist_ok=True)

# Kiválasztott videó tárolása
selected_video = None
# Kiválasztott elemzett videó tárolása
selected_analyzed_video = None

# Aszinkron futtatáshoz szükséges executor
executor = ThreadPoolExecutor()

# ----------------
# --- FUNKCIÓK ---
# ----------------

# Aszinkron függvény a videóelemzés futtatására háttérben
async def run_in_thread(func):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func)

# Aszinkron függvény a videóelemzés futtatására visszajelzéssel
async def run_pipeline_with_feedback(video_path: str, loader_dialog, container_to_hide):
    # Töltőképernyő mögötti konténer elrejtése
    container_to_hide.set_visibility(False)
    # Töltőképernyő megnyitása
    loader_dialog.open()
    # Elemzés futtatása a háttérben
    await run_in_thread(lambda: run_analysis_pipeline(video_path))
    # Töltőképernyő bezárása
    loader_dialog.close()
    ui.notify("Elemzés befejeződött!")
    ui.navigate.to('/analyzed_videos')

# Videófájl megnyitása a rendszer alapértelmezett lejátszójában
def open_video_file(path):
    try:
        os.startfile(Path(path).resolve())
    except Exception as e:
        ui.notify(f"Hiba a videó megnyitásakor: {e}", color="red")

# Videó feltöltése és mentése a megadott könyvtárba
def save_uploaded_video(e: events.UploadEventArguments):
    target_path = INPUT_DIR / e.name
    # Videó átmásolása a megadott könyvtárba
    with open(target_path, 'wb') as f:
        f.write(e.content.read())
    ui.notify(f"Videó feltöltve: {e.name}")

# Adott videófájl törlése gombnyomásra (feltöltött videók)
def delete_video_file(file_path: Path):
    if file_path.exists():
        file_path.unlink()
        ui.notify(f"Törölve: {file_path.name}")
        # Oldal frissítése a törlés után
        ui.navigate.to('/uploaded_videos')

# Adott elemzett videófájl részleteinek megjelenítése gombnyomásra
def show_analyzed_video_details(video_file: Path):
    global selected_analyzed_video
    selected_analyzed_video = video_file
    ui.navigate.to('/analyzed_video_detail')

# Adott elemzett videófájlhoz tartozó mappa törlése gombnyomásra
def delete_analyzed_video_file(video_path: Path):
    try:
        # Az elemzett videó mappájának törlése
        folder_path = video_path.parent
        shutil.rmtree(folder_path)
        ui.notify(f"Törölve: {folder_path.name}")
        ui.navigate.to('/analyzed_videos')
    except Exception as e:
        ui.notify(f"A fájlt nem lehet törölni, mert még használatban van!", color='orange')

# Videó kiválasztása
def select_video_for_analysis(video_file: Path):
    global selected_video
    selected_video = video_file
    ui.notify(f"Kiválasztott videó: {video_file.name}")
    ui.navigate.to('/analysis_config')

# ---------------
# --- OLDALAK ---
# ---------------

# --- Kezdőlap ("/") ---
@ui.page("/")
def start_page():
    with ui.column().classes("w-full h-screen justify-between absolute-center items-center"):
        
        # --- FELSŐ SZEKCIÓ ---
        with ui.column().classes("items-center pt-16 gap-8 mb-24"):
            ui.image("/logo/logo.png").classes("w-96 h-auto rounded shadow")

        with ui.column().classes("items-center gap-1"):
            ui.label("JasmInsight futballanalízis rendszer").classes("text-2xl font-extrabold mb-4")
            ui.markdown("*„A pályán minden mozdulat megmagyarázható. Csak elég sokáig kell figyelned, és tudnod kell, mit keress.*").classes("italic text-gray-300")
            ui.markdown("Marcelo Bielsa").classes("text-medium text-gray-300")

        # --- KÖZÉPSŐ GOMBOK ---
        with ui.column().classes("items-center justify-center").style("flex-grow: 1"):
            with ui.column().classes("gap-4 w-64"):
                ui.button("Indítás", on_click=lambda: ui.navigate.to("/main_page")).classes("w-full")
                ui.button("Kilépés", on_click=app.shutdown).classes("w-full mb-6")

        # --- ALSÓ SZEKCIÓ ---
        with ui.column().classes("items-center pb-24 gap-1"):
            ui.markdown("Szakdolgozati projekt").classes("text-sm")
            ui.markdown("Széchenyi István Egyetem, Mérnökinformatikus BSc szak").classes("text-sm mb-8")
            ui.markdown("https://github.com/pappakos0403/football_analysis").classes("text-sm text-gray-400 italic mb-2")
            ui.markdown("Build v1.2 • 2025. június").classes("text-sm text-gray-400 italic")

# --- Feltöltött videók oldala ("/uploaded_videos") ---
@ui.page("/uploaded_videos")
def uploaded_videos_page():
    with ui.column().classes("absolute-center items-center gap-4"):
        # Feltöltött videók oldalának címe
        ui.label("Feltöltött videók").classes("text-2xl font-semibold")
        # Feltöltött videók mappájának ellenőrzése
        video_files = list(INPUT_DIR.glob("*.mp4")) + list(INPUT_DIR.glob("*.avi")) + list(INPUT_DIR.glob("*.mkv"))
        if not video_files:
            ui.label("A Feltöltött videók mappája üres!").classes("text-white")
        else:
            with ui.row().classes("justify-center items-start flex-wrap gap-8"):
                for video_file in video_files:
                    with ui.column().classes("items-center"):
                        # Thumbnail + videólejátszó
                        ui.video(f'/videos/{video_file.name}').props('controls').classes("w-64 h-36 rounded shadow")
                        # Videók címei
                        ui.label(video_file.name).classes("text-sm text-center text-white mt-2")
                        # Videó törlése gomb
                        ui.button(on_click=lambda f=video_file: delete_video_file(f))\
                            .props('color="red" icon="delete"')
        # Visszalépés a menübe
        ui.button("Vissza a menübe", on_click=lambda: ui.navigate.to("/main_page")).classes("mt-4")

# --- Elemzett videók oldala ("/analyzed_videos") ---
@ui.page("/analyzed_videos")
def analyzed_videos_page():
    with ui.column().classes("absolute-center items-center gap-4"):
        ui.label("Elemzett videók").classes("text-2xl font-semibold")
        video_dirs = [p for p in OUTPUT_DIR.iterdir() if p.is_dir() and p.name.startswith("annotated_")]

        if not video_dirs:
            ui.label("Az elemzett videók mappája üres!").classes("text-white")
        else:
            with ui.row().classes("justify-center flex-wrap gap-8"):
                for video_dir in video_dirs:
                    video_name = video_dir.name.replace("annotated_", "")
                    video_path = video_dir / f"annotated_{video_name}.mp4"
                    with ui.column().classes("items-center cursor-pointer").on("click", lambda f=video_path: show_analyzed_video_details(f)):
                        ui.icon("folder").classes("text-yellow-400 text-6xl")
                        ui.label(video_name + ".mp4").classes("text-sm text-center text-white mt-2 max-w-48 truncate")

        ui.button("Vissza", on_click=lambda: ui.navigate.to("/main_page")).classes("mt-4")

@ui.page("/analyzed_video_detail")
def analyzed_video_detail_page():
 
    with ui.column().classes("absolute-center items-center gap-4"):
        # Statikus előnézet
        ui.label("Elemzett videó részletei").classes("text-2xl font-semibold")

        # Elemzett videó thumbnailjének megjelenítése
        thumbnail_path = f"/analyzed_videos/{selected_analyzed_video.parent.name}/thumbnail/thumbnail.jpg"
        ui.image(thumbnail_path).classes("w-96 h-56 rounded shadow object-cover")

        # Elemzett videó neve
        ui.label(selected_analyzed_video.name).classes("text-white text-center")

        # Gombok
        with ui.column().classes("gap-2 w-64"):
            ui.button("Videó megnyitása", on_click=lambda: open_video_file(selected_analyzed_video)).classes("w-full")
            ui.button("Hőtérképek", on_click=lambda: ui.navigate.to("/heatmaps")).classes("w-full")
            ui.button("Statisztikák és grafikonok", on_click=lambda: ui.navigate.to("/statistics")).classes("w-full")
            ui.button("Törlés", on_click=lambda: delete_analyzed_video_file(selected_analyzed_video)).props('color="red" icon="delete"').classes("w-full")

        # Vissza gomb
        ui.button("Vissza az elemzett videókhoz", on_click=lambda: ui.navigate.to("/analyzed_videos")).classes("mt-6")

@ui.page("/heatmaps")
def heatmaps_page():
    with ui.column().classes("absolute-center items-center gap-4"):
        # Oldalcím
        ui.label("Hőtérképek").classes("text-2xl font-semibold")

        # Hőtérképek könyvtárak definiálása
        heatmap_dir = selected_analyzed_video.parent / "heatmaps"
        team1_dir = heatmap_dir / "team1"
        team2_dir = heatmap_dir / "team2"
        ball_dir = heatmap_dir / "ball_heatmap"

        # Kategorizált fájlgyűjtés
        categorized = OrderedDict({
            "Team1 hőtérképei": sorted(team1_dir.glob("*.png")) if team1_dir.exists() else [],
            "Team2 hőtérképei": sorted(team2_dir.glob("*.png")) if team2_dir.exists() else [],
            "Labda hőtérképe": sorted(ball_dir.glob("*.png")) if ball_dir.exists() else [],
        })

        # Ha egyik kategóriában sincs fájl
        if not any(categorized.values()):
            ui.label("Nem találhatóak hőtérképek ehhez az elemzett videóhoz.").classes("text-white")
            ui.button("Vissza", on_click=lambda: ui.navigate.to("/analyzed_video_detail")).classes("mt-6")
            return

        # Dialógus és képállapot
        image_dialog = ui.dialog().props('maximized').classes("bg-black bg-opacity-90")
        current_index = {"value": 0}
        flat_file_list = []

        with image_dialog:
            with ui.column().classes("items-center relative pt-12"):
                # Bezáró gomb
                ui.button('X', on_click=image_dialog.close).props('flat').classes(
                    'absolute right-4 top-4 z-50 bg-red-600 text-white font-bold w-8 h-8 rounded flex items-center justify-center'
                )
                # Nagy kép megjelenítése
                image_display = ui.image().classes("w-full max-w-screen-xl h-auto object-contain rounded shadow")

                # Lapozógombok
                with ui.row().classes("w-full justify-center items-center gap-8 p-4 bg-black bg-opacity-70"
                                      ).style('position: absolute; bottom: 0;'):
                    ui.button("⬅️", on_click=lambda: show_image(current_index["value"] - 1)).classes("w-24")
                    ui.button("➡️", on_click=lambda: show_image(current_index["value"] + 1)).classes("w-24")

        # Kép megjelenítő függvény
        def show_image(index: int):
            if 0 <= index < len(flat_file_list):
                current_index["value"] = index
                rel_path = f"/analyzed_videos/{flat_file_list[index].relative_to('output_videos').as_posix()}"
                image_display.set_source(rel_path)
                image_dialog.open()

        # Kategóriánkénti képkártyák megjelenítése
        for category, files in categorized.items():
            if not files:
                continue

            # Kategória címe
            ui.label(category).classes("text-xl font-semibold mt-6 text-white")

            # Képkártyák sorban
            with ui.row().classes("justify-center flex-wrap gap-4 max-w-screen-2xl"):
                for idx, file_path in enumerate(files):
                    flat_index = len(flat_file_list)
                    flat_file_list.append(file_path)
                    rel_path = f"/analyzed_videos/{file_path.relative_to('output_videos').as_posix()}"
                    with ui.column().classes("items-center cursor-pointer"):
                        with ui.card().classes("p-2 hover:bg-gray-700 transition-colors duration-200") \
                                .on('click', lambda i=flat_index: show_image(i)):
                            ui.image(rel_path).classes("w-40 h-28 rounded shadow object-cover")
                            ui.label(file_path.name).classes("text-xs text-white text-center mt-1 max-w-40 truncate")

        # Vissza gomb
        ui.button("Vissza", on_click=lambda: ui.navigate.to("/analyzed_video_detail")).classes("mt-6")

# --- Statisztikák és grafikonok oldal ("/statistics") ---
@ui.page("/statistics")
def statistics_page():

    from player_statistics import load_player_stats

    # Játékosonkénti statisztikák
    statistics_dir = selected_analyzed_video.parent / "statistics"
    stats_path = statistics_dir / "player_stats.json"
    player_stats = load_player_stats(stats_path)

    # Játékosonkénti statisztikák dialógusa
    player_dialog = ui.dialog().props('persistent').classes("bg-gray-900 text-white")
    with player_dialog:
        with ui.column().classes("items-center gap-3"):
            dialog_title = ui.label("").classes("text-xl font-bold")

            jersey_image = ui.image().classes("w-48 h-auto")

            stat_container = ui.column().classes("gap-3 text-sm items-start")

            # Bezárás gomb
            ui.button("Bezárás", on_click=player_dialog.close).classes("w-40 mt-8")

    HUNGARIAN_LABELS = {
        "presence_ratio": ("📊", "Detektálási arány"),
        "distance_m": ("📏", "Megtett távolság (m)"),
        "avg_speed_kmh": ("🚶", "Átlagsebesség (km/h)"),
        "max_speed_kmh": ("🏃", "Max sebesség (km/h)"),
        "accurate_passes": ("✅", "Pontos passzok száma"),
        "inaccurate_passes": ("❌", "Pontatlan passzok száma"),
        "ball_possession_count": ("⚽", "Labdabirtoklások száma"),
        "offside_time": ("🚩", "Lesen töltött idő")
    }

    # Játékosok statisztikáinak megjelenítése
    def show_player_stats(track_id: int):
        stats = player_stats.get(str(track_id))
        if not stats:
            return

        dialog_title.set_text(f"Játékos #{track_id} statisztikái")
        stat_container.clear()

        # Mez kép beállítása
        jersey_path = statistics_dir / f"team{stats['team']}shirt_images" / f"{track_id}.png"
        if jersey_path.exists():
            jersey_image.set_source(jersey_path.as_posix())
        else:
            jersey_image.set_source("")

        with stat_container:
            ui.table(
                columns=[
                    {"name": "stat", "label": "Mutató", "field": "stat", "align": "center", "style": "width: 300px;"},
                    {"name": "value", "label": "Érték", "field": "value", "align": "center", "style": "width: 300px;"}
                ],
                rows=[
                    {"stat": f"{icon} {label}", "value": value}
                    for key, value in stats.items()
                    if key not in ("track_id", "team")
                    for icon, label in [HUNGARIAN_LABELS.get(key, ("", key))]
                ],
                row_key="stat"
            ).classes("w-full text-sm text-white bg-gray-800").style("border-radius: 8px; text-align: center;")
        
        player_dialog.open()

    with ui.column().classes("w-full max-w-screen-xl mx-auto px-4 py-6 gap-4 items-center"):
        ui.label("Statisztikák és grafikonok").classes("text-2xl font-semibold")

        # Statisztikák mappa betöltése az aktuális elemzett videó alapján
        statistics_dir = selected_analyzed_video.parent / "statistics"
        stat_files = sorted(statistics_dir.glob("*.png"))

        if not stat_files:
            ui.label("Nem található statisztika vagy grafikon ehhez az elemzett videóhoz.").classes("text-white")
            ui.button("Vissza", on_click=lambda: ui.navigate.to("/analyzed_video_detail")).classes("mt-6")
            return

        # Kategóriák szerinti csoportosítás fájlnév alapján
        categorized = defaultdict(list)
        for file in stat_files:
            name = file.name.lower()
            if "activity" in name:
                categorized["Játékosok aktivitási statisztikái"].append(file)
            elif "players_per_half" in name:
                categorized["Játékosok elhelyezkedése a térfeleken"].append(file)
            elif "offside" in name:
                categorized["Lesek statisztikái"].append(file)
            else:
                categorized["Egyéb statisztikák"].append(file)

        # Nagy kép dialógus és állapot
        image_dialog = ui.dialog().props('maximized').classes("bg-black bg-opacity-90")
        current_index = {"value": 0}
        flat_file_list = []

        with image_dialog:
            with ui.column().classes("items-center relative pt-12"):
                # Bezárás gomb
                ui.button('X', on_click=image_dialog.close).props('flat').classes(
                    'absolute right-4 top-4 z-50 bg-red-600 text-white font-bold w-8 h-8 rounded flex items-center justify-center'
                )
                # Nagy kép
                image_display = ui.image().classes("w-full max-w-screen-xl h-auto object-contain rounded shadow")

                # Lapozógombok
                with ui.row().classes("w-full justify-center items-center gap-8 p-4 bg-black bg-opacity-70"
                                      ).style('position: absolute; bottom: 0;'):
                    ui.button("⬅️", on_click=lambda: show_image(current_index["value"] - 1)).classes("w-24")
                    ui.button("➡️", on_click=lambda: show_image(current_index["value"] + 1)).classes("w-24")

        # Kép megjelenítő függvény
        def show_image(index: int):
            if 0 <= index < len(flat_file_list):
                current_index["value"] = index
                rel_path = f"/analyzed_videos/{flat_file_list[index].relative_to('output_videos').as_posix()}"
                image_display.set_source(rel_path)
                image_dialog.open()

        # Kategóriánkénti előnézeti képek megjelenítése
        for category, files in categorized.items():
            if not files:
                continue

            # Kategória címe
            ui.label(category).classes("text-xl font-semibold mt-6 text-white")

            # Kép előnézetek
            with ui.row().classes("justify-center flex-wrap gap-4 max-w-screen-2xl"):
                for idx, stat_path in enumerate(files):
                    flat_index = len(flat_file_list)
                    flat_file_list.append(stat_path)
                    rel_path = f"/analyzed_videos/{stat_path.relative_to('output_videos').as_posix()}"
                    with ui.column().classes("items-center cursor-pointer"):
                        with ui.card().classes("p-2 hover:bg-gray-700 transition-colors duration-200") \
                                .on('click', lambda i=flat_index: show_image(i)):
                            ui.image(rel_path).classes("w-40 h-28 rounded shadow object-cover")
                            ui.label(stat_path.name).classes("text-xs text-white text-center mt-1 max-w-40 truncate")

        # Játékosonkénti statisztikák megjelenítése csoportosítva
        shirt_dir_base = selected_analyzed_video.parent / "statistics"
        team1_dir = shirt_dir_base / "team1shirt_images"
        team2_dir = shirt_dir_base / "team2shirt_images"

        def render_jersey_cards(title: str, jersey_dir: Path):
            jersey_files = sorted(jersey_dir.glob("*.png"))
            if not jersey_files:
                return
            ui.label(title).classes("text-xl font-semibold mt-6 text-white")

            with ui.row().classes("justify-center flex-wrap gap-4 max-w-screen-xl"):
                for jersey_file in jersey_files:
                    flat_index = len(flat_file_list)
                    flat_file_list.append(jersey_file)
                    rel_path = f"/analyzed_videos/{jersey_file.relative_to('output_videos').as_posix()}"

                    with ui.column().classes("items-center cursor-pointer"):
                        with ui.card().classes("p-2 hover:bg-gray-700 transition-colors duration-200") \
                                .on('click', lambda tid=int(jersey_file.stem): show_player_stats(tid)):
                            with ui.column().classes("items-center"):
                                ui.image(rel_path).classes("w-32 h-auto object-contain")
                                ui.label(jersey_file.stem).classes("text-xs text-white text-center mt-1")

        if team1_dir.exists():
            render_jersey_cards("Team1 játékosai", team1_dir)
        if team2_dir.exists():
            render_jersey_cards("Team2 játékosai", team2_dir)

        # Összehasonlítás gomb
        ui.button("Összehasonlítás").classes("mt-6")

        # Vissza gomb
        ui.button("Vissza", on_click=lambda: ui.navigate.to("/analyzed_video_detail")).classes("mt-6")

# --- Videó kiválasztása elemzéshez ("/select_video_for_analysis") ---
@ui.page("/select_video_for_analysis")
def select_video_for_analysis_page():
    with ui.column().classes("absolute-center items-center gap-4"):
        # Videó kiválasztása oldal címe
        ui.label("Videó kiválasztása elemzéshez").classes("text-2xl font-semibold")
        ui.markdown("Kattintson a videóra a kiválasztáshoz!").classes("text-center")
        
        # Feltöltött videók mappájának ellenőrzése
        video_files = list(INPUT_DIR.glob("*.mp4")) + list(INPUT_DIR.glob("*.avi")) + list(INPUT_DIR.glob("*.mkv"))
        if not video_files:
            ui.label("Nincs feltöltött videó! Először töltsön fel videókat.").classes("text-orange-400 text-center")
            # Videó feltöltése gomb
            ui.button("Videó feltöltése", on_click=lambda: ui.navigate.to("/main_page")).classes("mt-4")
        else:
            with ui.row().classes("justify-center items-start flex-wrap gap-8 max-w-4xl"):
                for video_file in video_files:
                    with ui.column().classes("items-center cursor-pointer"):
                        # Kattintható videó container
                        with ui.card().classes("p-4 hover:bg-gray-700 transition-colors duration-200") \
                            .on('click', lambda f=video_file: select_video_for_analysis(f)):
                            # Videó thumbnail (első frame előnézet) - letiltott vezérlőkkel
                            ui.video(f'/videos/{video_file.name}').classes("w-64 h-36 rounded shadow") \
                                .props('muted preload="metadata" style="pointer-events: none;"')
                            # Videók címei
                            ui.label(video_file.name).classes("text-sm text-center text-white mt-2 max-w-64 truncate")
        
        # Visszalépés az elemzés konfigurációs oldalra
        ui.button("Vissza", on_click=lambda: ui.navigate.to("/analysis_config")).classes("mt-4")

# --- Elemzés konfigurációs oldal ("/analysis_config") ---
@ui.page("/analysis_config")
def analysis_config_page():
    page_container = ui.column().classes("absolute-center items-center gap-4")

    # Státusz frissítése
    def update_status(text: str):
        status_label.set_text(text) # A .set_text() használata egyértelműbbé teszi a szándékot

    # Aszinkron függvény a videóelemzés futtatására visszajelzéssel
    async def run_pipeline_with_feedback(video_path: str, loader_dialog, container_to_hide, config: dict):
        # Töltőképernyő mögötti konténer elrejtése
        container_to_hide.set_visibility(False)
        # Töltőképernyő megnyitása
        loader_dialog.open()
        # Elemzés futtatása a háttérben a helyi frissítő függvénnyel
        await run_in_thread(lambda: run_analysis_pipeline(video_path, update_status, config))
        # Töltőképernyő bezárása
        loader_dialog.close()
        ui.notify("Elemzés befejeződött!")
        ui.navigate.to('/analyzed_videos')

    with page_container:
        # Elemzés konfigurációs oldal címe
        ui.label("Videó elemzése").classes("text-2xl font-semibold")

        # Kiválasztott videó megjelenítése
        if selected_video:
            ui.markdown(f"**Kiválasztott videó:** {selected_video.name}").classes("text-center text-green-400")
        else:
            ui.markdown("Még nincs videó kiválasztva az elemzéshez.").classes("text-center text-orange-400")

        # Videó kiválasztása gomb
        ui.button("Videó kiválasztása",
                 on_click=lambda: ui.navigate.to("/select_video_for_analysis")).classes("w-48 text-lg")

        # Töltőképernyő dialógus létrehozása
        loader_dialog = ui.dialog().classes("bg-transparent")
        with loader_dialog:
            with ui.column().classes("items-center gap-4"):
                ui.spinner(size="lg", color="primary")
                ui.label("Elemzés folyamatban...").classes("text-white text-lg text-center")
                # A státusz címke lokális létrehozása
                status_label = ui.label("").classes("text-white text-lg text-center")

        # Elemzés indítása gomb (csak ha van kiválasztott videó)
        if selected_video:
            advanced_dialog = ui.dialog().props('modal').classes("bg-black bg-opacity-80 z-50")
            with advanced_dialog:
                with ui.column().classes("p-4 gap-2 items-center"):
                    with ui.column().classes("items-center text-center"):
                        # Panel címe és leírása
                        ui.label("Elemzés konfigurálása").classes("text-2xl font-bold mb-2")
                        ui.markdown("Válassza ki a kívánt opciókat az elemzéshez!").classes("text-sm mb-4")

                    # Checkboxok külön konténerben, balra igazítva
                    with ui.row().classes("w-full gap-8").style("display: flex; justify-content: space-between;"):
                        # Bal oldali oszlop, annotációs opciók
                        with ui.column().classes("gap-2").style("flex: 1; min-width: 0;"):
                            with ui.row().classes("justify-center w-full"):
                                ui.label("Annotációs opciók").classes("text-lg font-bold mb-2 text-yellow")
                            show_player_ellipses_cb = ui.checkbox("Játékosok annotálása", value=True)
                            show_player_ids_cb = ui.checkbox("Játékosok azonosítói", value=True)
                            show_referees_cb = ui.checkbox("Játékvezetők annotálása", value=True)
                            show_ball_triangle_cb = ui.checkbox("Labda annotálása", value=True)
                            show_team_colors_topbar_cb = ui.checkbox("Csapatokat jelölő négyzetek", value=True)
                            show_speed_distance_cb = ui.checkbox("Sebesség és megtett távolság annotálása", value=True)
                            show_possession_overlay_cb = ui.checkbox("Labdabirtoklás overlay", value=True)
                            show_pass_statistics_cb = ui.checkbox("Passz statisztikák overlay", value=True)
                            show_closest_player_triangle_cb = ui.checkbox("Legközelebbi játékos a labdához annotálása", value=True)
                            show_offside_flags_cb = ui.checkbox("Lesek annotálása", value=True)
                            show_keypoints_cb = ui.checkbox("Kulcspontok annotálása", value=False)
                            show_player_coordinates_cb = ui.checkbox("Játékosok koordinátáinak annotálása", value=False)

                        # Jobb oldali oszlop, statisztikák és grafikonok opciói
                        with ui.column().classes("gap-2").style("flex: 1; min-width: 0;"):
                            with ui.row().classes("justify-center w-full"):
                                ui.label("Statisztikák és grafikonok").classes("text-lg font-bold mb-2 text-yellow")
                            show_player_activity_stats_cb = ui.checkbox("Játékos aktivitás statisztikák", value=True)
                            show_players_per_half_graph_cb = ui.checkbox("Játékosok eloszlása térfélenként", value=True)
                            show_offside_stats_cb = ui.checkbox("Lesek statisztikai ábrája", value=True)
                            show_player_heatmaps_cb = ui.checkbox("Játékos hőtérképek", value=True)
                            show_ball_heatmap_cb = ui.checkbox("Labda hőtérképe", value=True)
                    ui.button("Bezárás", on_click=advanced_dialog.close).classes("mt-4")

            with ui.column().classes("items-center gap-2"):
                ui.button("Elemzés indítása",
                        # A gomb a helyileg definiált futtató függvényt hívja meg
                        on_click=lambda: run_pipeline_with_feedback(
                            str(selected_video), 
                            loader_dialog, 
                            page_container,
                            {
                                # Annotálás a kimeneti videón
                                "show_player_ellipses": show_player_ellipses_cb.value,
                                "show_player_ids": show_player_ids_cb.value,
                                "show_referees": show_referees_cb.value,
                                "show_ball_triangle": show_ball_triangle_cb.value,
                                "show_team_colors_topbar": show_team_colors_topbar_cb.value,
                                "show_speed_distance": show_speed_distance_cb.value,
                                "show_possession_overlay": show_possession_overlay_cb.value,
                                "show_pass_statistics": show_pass_statistics_cb.value,
                                "show_closest_player_triangle": show_closest_player_triangle_cb.value,
                                "show_offside_flags": show_offside_flags_cb.value,
                                "show_keypoints": show_keypoints_cb.value,
                                "show_player_coordinates": show_player_coordinates_cb.value,
                                # Statisztikák és grafikonok
                                "show_player_activity_stats": show_player_activity_stats_cb.value,
                                "show_players_per_half_graph": show_players_per_half_graph_cb.value,
                                "show_offside_stats": show_offside_stats_cb.value,
                                "show_player_heatmaps": show_player_heatmaps_cb.value,
                                "show_ball_heatmap": show_ball_heatmap_cb.value
                            })
                        ).classes("w-48 text-lg mt-2")
                # Elemzés konfigurációs beállítások
                ui.button("Haladó beállítások", on_click=advanced_dialog.open).classes("w-48 text-lg mt-2")
            

        # Vissza a menübe gomb -> visszalépés a főoldalra
        ui.button("Vissza", on_click=lambda: ui.navigate.to("/main_page")).classes("mt-4")

# --- Elemezni kívánt videó kiválasztása oldal ---
@ui.page("/select_video")
def select_video_page():
    with ui.column().classes("absolute-center items-center gap-4"):
        # Videó kiválasztása oldal címe
        ui.label("Videó kiválasztása").classes("text-2xl font-semibold")
        # Videó kiválasztása gomb --> videó kiválasztása oldalra navigál
        ui.button("Videó kiválasztása", on_click=lambda: ui.navigate.to("/analysis_config")).classes("w-48 text-lg")
        # Vissza az elemzés konfigurációs oldalra --> elemzés konfigurációs oldalra navigál
        ui.button("Vissza", on_click=lambda: ui.navigate.to("/main_page")).classes("mt-4")

# --- Teszt oldal ("/teszt") ---
@ui.page("/main_page")
def main_page():
    with ui.column().classes("absolute-center items-center"):
        # Teszt oldal címe
        ui.label("Videó elemzése").classes("text-2xl font-semibold")
        # Videó feltöltése
        ui.upload(on_upload=save_uploaded_video,
                  multiple=True, 
                  on_rejected=lambda e: ui.notify("Csak mp4/avi kiterjesztésű videók engedélyezettek!")
                  ).props('accept=".mp4,.avi,.mkv" auto-upload')  
        # Videó feltöltésének leírása
        ui.markdown("Itt töltheti fel az elemezni kívánt videót (mp4/avi/mkv).").classes("text-center")
        with ui.column().classes("gap-2 w-64"):
            # Feltöltött videók -> feltöltött videók oldalára navigál
            ui.button("Feltöltött videók", on_click=lambda: ui.navigate.to("/uploaded_videos")).classes("w-full")
            # Elemzés konfigurációs oldal -> elemzés konfigurációs oldalra navigál
            ui.button("Videóelemzés", on_click=lambda: ui.navigate.to("/analysis_config")).classes("w-full")
            # Elemzett videók -> elemzett videók oldalára navigál
            ui.button("Elemzett videók", on_click=lambda: ui.navigate.to("/analyzed_videos")).classes("w-full")
            # Vissza a kezdőlapra gomb -> kezdőlapra navigál
            ui.button("Vissza a kezdőlapra", on_click=lambda: ui.navigate.to("/")).classes("w-full")

# --- GUI indítása natív módban, teljes képernyőn ---
app.add_static_files('/videos', INPUT_DIR)
app.add_static_files('/analyzed_videos', OUTPUT_DIR)
app.add_static_files('/logo', 'logo')
ui.run(native=True, fullscreen=True, dark=True, title="JasmInsight")