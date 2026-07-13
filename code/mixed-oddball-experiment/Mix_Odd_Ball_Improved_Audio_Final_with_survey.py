from psychopy import prefs
prefs.hardware["audioLib"] = ["PTB"]
prefs.hardware["audioLatencyMode"] = 3

from psychopy import visual, core, event, sound, gui, data
import random
import csv
import os
import numpy as np
import tkinter as tk
from datetime import datetime, timezone

FULLSCREEN = True
WINDOW_SIZE = (1200, 800)
BG_COLOR = "black"
TEXT_COLOR = "white"
RESPONSE_KEY = "space"
QUIT_KEY = "escape"
EN_FONT = "Arial"
HI_FONT = "Arial"

BASELINE_REST_DUR = 300.0
ODD_STIM_DURATION = 0.150
ODD_TOTAL_TRIAL_DURATION = 1.500
WM_BLOCKS = 15
WM_ENCODE_ITEM_DUR = 0.75
WM_BLANK_DUR = 0.25
WM_FEEDBACK_DUR = 0.6
RT_TRIALS = 30
RT_FLASH_DUR = 0.100
RT_RESP_WINDOW = 1.2

STANDARD_FREQ = 500
HIGH_FREQ = 1000
BEEP_FREQ = 800
TONE_DURATION = ODD_STIM_DURATION
BEEP_DURATION = 0.250
TONE_SAMPLE_RATE = 48000
TONE_VOLUME = 0.8
TONE_RAMP_SEC = 0.005

PHASE1_COUNTS = {
    "visual_std_audio_none": 18,
    "visual_std_audio_std": 15,
    "visual_std_audio_odd": 3,
    "visual_odd_audio_none": 5,
    "visual_odd_audio_std": 4,
    "visual_odd_audio_odd": 0,
}
PHASE2_COUNTS = {
    "visual_std_audio_none": 14,
    "visual_std_audio_std": 12,
    "visual_std_audio_odd": 5,
    "visual_odd_audio_none": 5,
    "visual_odd_audio_std": 5,
    "visual_odd_audio_odd": 4,
}
PHASE3_COUNTS = {
    "visual_std_audio_none": 16,
    "visual_std_audio_std": 12,
    "visual_std_audio_odd": 8,
    "visual_odd_audio_none": 7,
    "visual_odd_audio_std": 7,
    "visual_odd_audio_odd": 10,
}

MARKERS = {
    "section_start_baseline": 90,
    "section_end_baseline": 91,
    "section_start_dass21": 80,
    "section_end_dass21": 81,
    "dass21_response": 82,
    "section_start_oddball": 100,
    "section_end_oddball": 101,
    "section_start_wm": 110,
    "section_end_wm": 111,
    "section_start_rt": 120,
    "section_end_rt": 121,
    "survey_start": 140,
    "survey_difficulty": 141,
    "survey_stress_relax": 142,
    "survey_focus": 143,
    "survey_mental_effort": 144,
    "survey_fatigue": 145,
    "survey_end": 146,
    "visual_std_audio_none": 201,
    "visual_std_audio_std": 202,
    "visual_std_audio_odd": 203,
    "visual_odd_audio_none": 204,
    "visual_odd_audio_std": 205,
    "visual_odd_audio_odd": 206,
    "wm_encode": 300,
    "wm_probe": 301,
    "wm_response": 302,
    "rt_flash": 400,
}

DASS21_ITEMS = [
    (1, "stress", "I found it hard to wind down.", "मुझे शांत होना कठिन लगा।"),
    (2, "anxiety", "I was aware of dryness of my mouth.", "मुझे अपने मुँह में सूखापन महसूस हुआ।"),
    (3, "depression", "I could not seem to experience any positive feeling at all.", "मुझे कोई भी सकारात्मक भावना महसूस नहीं हो रही थी।"),
    (4, "anxiety", "I experienced breathing difficulty.", "मुझे साँस लेने में कठिनाई महसूस हुई।"),
    (5, "depression", "I found it difficult to work up the initiative to do things.", "मुझे काम शुरू करने की पहल करना कठिन लगा।"),
    (6, "stress", "I tended to over-react to situations.", "मैं परिस्थितियों पर जरूरत से ज्यादा प्रतिक्रिया दे रहा/रही था/थी।"),
    (7, "anxiety", "I experienced trembling.", "मुझे कंपकंपी महसूस हुई।"),
    (8, "stress", "I felt that I was using a lot of nervous energy.", "मुझे लगा कि मैं बहुत अधिक मानसिक तनाव/ऊर्जा खर्च कर रहा/रही था/थी।"),
    (9, "anxiety", "I was worried about situations in which I might panic and make a fool of myself.", "मुझे ऐसी स्थितियों की चिंता थी जिनमें मैं घबरा सकता/सकती था/थी और शर्मिंदा हो सकता/सकती था/थी।"),
    (10, "depression", "I felt that I had nothing to look forward to.", "मुझे लगा कि मेरे पास आगे देखने/उम्मीद करने के लिए कुछ नहीं है।"),
    (11, "stress", "I found myself getting agitated.", "मैं स्वयं को बेचैन या उत्तेजित महसूस कर रहा/रही था/थी।"),
    (12, "stress", "I found it difficult to relax.", "मुझे आराम करना कठिन लगा।"),
    (13, "depression", "I felt down-hearted and blue.", "मैं उदास और निराश महसूस कर रहा/रही था/थी।"),
    (14, "stress", "I was intolerant of anything that kept me from getting on with what I was doing.", "जो चीजें मेरे काम में बाधा डालती थीं, उन्हें सहन करना मुझे कठिन लगा।"),
    (15, "anxiety", "I felt I was close to panic.", "मुझे लगा कि मैं घबराहट के बहुत करीब था/थी।"),
    (16, "depression", "I was unable to become enthusiastic about anything.", "मैं किसी भी चीज के लिए उत्साहित नहीं हो पा रहा/रही था/थी।"),
    (17, "depression", "I felt I was not worth much as a person.", "मुझे लगा कि एक व्यक्ति के रूप में मेरा अधिक मूल्य नहीं है।"),
    (18, "stress", "I felt that I was rather touchy.", "मुझे लगा कि मैं बहुत जल्दी चिड़चिड़ा/चिड़चिड़ी हो रहा/रही था/थी।"),
    (19, "anxiety", "I was aware of the action of my heart in the absence of physical exertion.", "बिना शारीरिक मेहनत के भी मुझे अपने हृदय की धड़कन महसूस हो रही थी।"),
    (20, "anxiety", "I felt scared without any good reason.", "मुझे बिना किसी स्पष्ट कारण के डर लगा।"),
    (21, "depression", "I felt that life was meaningless.", "मुझे लगा कि जीवन अर्थहीन है।"),
]

DASS_OPTIONS = [
    "0\nDid not apply to me at all\nबिल्कुल लागू नहीं हुआ",
    "1\nApplied to me to some degree\nकुछ हद तक लागू हुआ",
    "2\nApplied to me considerably\nकाफी हद तक लागू हुआ",
    "3\nApplied to me very much\nबहुत अधिक लागू हुआ",
]


def send_marker(code):
    pass


def laptop_time_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def clear_experiment_screen():
    try:
        win.color = BG_COLOR
        win.flip()
    except NameError:
        pass


def make_tk_fullscreen(root):
    root.configure(bg="black")
    root.attributes("-topmost", True)
    root.attributes("-fullscreen", True)
    root.update_idletasks()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.geometry("%dx%d+0+0" % (screen_w, screen_h))
    root.lift()
    root.focus_force()
    return screen_w, screen_h


def make_tone_wave(freq_hz, duration_s, sample_rate, volume, ramp_sec):
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    wave = np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)
    ramp_n = int(ramp_sec * sample_rate)
    if 0 < ramp_n < (n_samples // 2):
        ramp = np.linspace(0.0, 1.0, ramp_n, dtype=np.float32)
        wave[:ramp_n] *= ramp
        wave[-ramp_n:] *= ramp[::-1]
    wave *= volume
    return np.column_stack((wave, wave)).astype(np.float32)


def classify_trial(visual_rare, audio_present, audio_rare):
    if visual_rare and audio_present and audio_rare:
        return "both_oddball"
    if visual_rare:
        return "visual_only"
    if audio_present and audio_rare:
        return "audio_only"
    return "standard_standard"


def condition_to_trial(condition_label):
    visual_rare = condition_label.startswith("visual_odd")
    audio_present = not condition_label.endswith("audio_none")
    audio_rare = condition_label.endswith("audio_odd")
    if not audio_present:
        audio_type = "none"
        tone_name = "none"
        audio_target = 0
    elif audio_rare:
        audio_type = "oddball"
        tone_name = "high_tone"
        audio_target = 1
    else:
        audio_type = "standard"
        tone_name = "standard_tone"
        audio_target = 0
    return {
        "trial_type": classify_trial(visual_rare, audio_present, audio_rare),
        "condition_label": condition_label,
        "condition_code": MARKERS[condition_label],
        "visual_type": "oddball" if visual_rare else "standard",
        "audio_type": audio_type,
        "audio_present": int(audio_present),
        "shape_name": "red_triangle" if visual_rare else "blue_circle",
        "tone_name": tone_name,
        "visual_target": int(visual_rare),
        "audio_target": audio_target,
        "any_target": int(visual_rare or audio_target),
        "both_targets": int(visual_rare and audio_target),
    }


def build_oddball_trials_from_counts(counts_dict):
    trials = []
    for condition_label, n_trials in counts_dict.items():
        for _ in range(n_trials):
            trials.append(condition_to_trial(condition_label))
    random.shuffle(trials)
    return trials


def cleanup(win):
    try:
        standard_tone.stop(reset=True)
        high_tone.stop(reset=True)
        end_beep.stop(reset=True)
    except Exception:
        pass
    win.close()
    core.quit()


def make_textbox2(win, text, pos, size, font_name, letter_height=0.03,
                  color="white", alignment="left", padding=0.01):
    return visual.TextBox2(
        win=win,
        text=text,
        pos=pos,
        size=size,
        units="height",
        font=font_name,
        letterHeight=letter_height,
        color=color,
        alignment=alignment,
        editable=False,
        fillColor=None,
        borderWidth=0,
        padding=padding,
    )


def show_tk_bilingual(en_text, hi_text, wait_for_space=True, duration_s=None):
    clear_experiment_screen()
    root = tk.Tk()
    root.title("Instructions")
    screen_w, screen_h = make_tk_fullscreen(root)

    container = tk.Frame(root, bg="black", width=int(screen_w * 1.0), height=int(screen_h * 1.0))
    container.pack(expand=True, padx=80, pady=35)
    container.pack_propagate(False)
    container.grid_columnconfigure(0, weight=1)
    container.grid_columnconfigure(1, weight=1)
    container.grid_rowconfigure(0, weight=1)

    en_label = tk.Label(
        container,
        text=en_text,
        font=("Arial", 20),
        fg="white",
        bg="black",
        justify="center",
        wraplength=520,
        anchor="e",
    )
    en_label.grid(row=0, column=0, sticky="nsew", padx=(0, 18))

    hi_label = tk.Label(
        container,
        text=hi_text,
        font=("Arial", 20),
        fg="white",
        bg="black",
        justify="center",
        wraplength=520,
        anchor="w",
    )
    hi_label.grid(row=0, column=1, sticky="nsew", padx=(18, 0))

    def close_window(event_obj=None):
        root.quit()

    def quit_experiment(event_obj=None):
        root.destroy()
        cleanup(win)

    if wait_for_space:
        root.bind("<space>", close_window)
    root.bind("<Escape>", quit_experiment)

    if duration_s is not None:
        root.after(int(duration_s * 1000), close_window)

    root.mainloop()
    try:
        root.destroy()
    except Exception:
        pass


def draw_bilingual_page(en_text, hi_text):
    show_tk_bilingual(en_text, hi_text, wait_for_space=False, duration_s=0.05)


def wait_for_bilingual_start(en_text, hi_text):
    show_tk_bilingual(en_text, hi_text, wait_for_space=True)


def draw_single_text(text, pos, size, font_name, letter_height,
                     color="white", alignment="center"):
    if font_name == HI_FONT:
        stim = make_textbox2(
            win=win,
            text=text,
            pos=pos,
            size=size,
            font_name=font_name,
            letter_height=letter_height,
            color=color,
            alignment=alignment,
            padding=0.01,
        )
    else:
        stim = visual.TextStim(
            win,
            text=text,
            color=color,
            font=font_name,
            height=letter_height,
            wrapWidth=size[0],
            alignText=alignment,
            pos=pos,
        )
    stim.draw()
    return stim


def draw_image_text(text, pos, box_size, font_name, pixel_size=None,
                    color="white", align="center", max_width_px=None):
    letter_height = 0.032
    if pixel_size is not None:
        if pixel_size >= 80:
            letter_height = 0.10
        elif pixel_size >= 50:
            letter_height = 0.055
        elif pixel_size >= 40:
            letter_height = 0.040
        elif pixel_size >= 30:
            letter_height = 0.030
        else:
            letter_height = 0.022
    return draw_single_text(text, pos, box_size, font_name, letter_height, color, align)


def draw_fixation_and_wait(win, fixation, duration):
    fixation.draw()
    win.flip()
    core.wait(duration)


def stop_all_tones():
    standard_tone.stop(reset=True)
    high_tone.stop(reset=True)


def schedule_tone_on_flip(win, tone_stim):
    stop_all_tones()
    next_flip = win.getFutureFlipTime(clock="ptb")
    tone_stim.play(when=next_flip)


def make_blank_row():
    return {
        "participant": exp_info["participant"],
        "session": exp_info["session"],
        "date": exp_info["date"],
        "Age (in Years)": exp_info["Age (in Years)"],
        "Gender(M/F)": exp_info["Gender(M/F)"],
        "Meditation Experience (Months)": exp_info["Meditation Experience (Months)"],
        "Are you vegetarian(1)/ non-vegetarian(2)eggetarian(3)":  exp_info["Are you vegetarian(1)/ non-vegetarian(2)eggetarian(3)"],
        "Do you do any kind of intoxication": exp_info["Do you do any kind of intoxication"],
        "Do you drink coffee": exp_info["Do you drink coffee"],
        "Do you drink tea": exp_info["Do you drink tea"],
        "Hour of sleep (total in 24 hrs)": exp_info["Hour of sleep (total in 24 hrs)"],
        "Sleep Quality": exp_info["Sleep Quality"],
        "section": "",
        "trial_num": "",
        "subtrial_num": "",
        "phase": "",
        "trial_type": "",
        "condition_label": "",
        "condition_code": "",
        "visual_type": "",
        "audio_type": "",
        "audio_present": "",
        "shape_name": "",
        "tone_name": "",
        "visual_target": "",
        "audio_target": "",
        "any_target": "",
        "both_targets": "",
        "stim_onset_global": "",
        "stim_onset_laptop_time": "",
        "stim_onset_utc": "",
        "response_global": "",
        "response_key": "",
        "response_laptop_time": "",
        "response_utc": "",
        "event_summary": "",
        "rt": "",
        "responded": "",
        "accuracy": "",
        "response_outcome": "",
        "wm_sequence": "",
        "wm_length": "",
        "wm_correct": "",
        "rt_flash_delay": "",
        "section_end_global": "",
        "section_end_laptop_time": "",
        "section_end_utc": "",
        "marker_event_label": "",
        "marker_code": "",
        "survey_task": "",
        "survey_item": "",
        "survey_response": "",
        "task_order": "",
        "dass_item_num": "",
        "dass_subscale": "",
        "dass_response": "",
    }


def write_row(writer, overrides):
    row = make_blank_row()
    row.update(overrides)
    if row.get("stim_onset_laptop_time") and not row.get("stim_onset_utc"):
        row["stim_onset_utc"] = row["stim_onset_laptop_time"]
    if row.get("response_laptop_time") and not row.get("response_utc"):
        row["response_utc"] = row["response_laptop_time"]
    if row.get("section_end_laptop_time") and not row.get("section_end_utc"):
        row["section_end_utc"] = row["section_end_laptop_time"]
    writer.writerow(row)


def send_and_log_marker(writer, section_name, trial_num, marker_label, marker_code,
                        phase_name="marker_event", summary=""):
    send_marker(marker_code)
    marker_global = round(global_clock.getTime(), 6)
    marker_utc = laptop_time_str()
    row = {
        "section": section_name,
        "trial_num": trial_num,
        "phase": phase_name,
        "stim_onset_global": marker_global,
        "stim_onset_laptop_time": marker_utc,
        "event_summary": summary if summary else marker_label,
        "marker_event_label": marker_label,
        "marker_code": marker_code,
        "task_order": task_order_label,
    }
    if marker_label.startswith("section_end") or marker_label == "survey_end":
        row["section_end_global"] = marker_global
        row["section_end_laptop_time"] = marker_utc
    write_row(writer, row)


def make_button_page_stims(title_en, title_hi, prompt_en, prompt_hi, option_labels):
    stims = [
        visual.TextStim(win, text=title_en, color=TEXT_COLOR, font=EN_FONT, height=0.034,
                        wrapWidth=0.62, alignText="left", anchorHoriz="left", pos=(-0.68, 0.31)),
        make_textbox2(win, title_hi, (0.36, 0.31), (0.60, 0.12), HI_FONT, 0.030, TEXT_COLOR, "left"),
        visual.TextStim(win, text=prompt_en, color=TEXT_COLOR, font=EN_FONT, height=0.028,
                        wrapWidth=0.62, alignText="left", anchorHoriz="left", pos=(-0.68, 0.14)),
        make_textbox2(win, prompt_hi, (0.36, 0.14), (0.60, 0.20), HI_FONT, 0.026, TEXT_COLOR, "left"),
    ]
    buttons = []
    if len(option_labels) == 4:
        x_positions = [-0.45, -0.15, 0.15, 0.45]
        button_width = 0.28
    else:
        x_positions = [-0.52, -0.26, 0.0, 0.26, 0.52]
        button_width = 0.22
    for i, option_text in enumerate(option_labels):
        rect = visual.Rect(win, width=button_width, height=0.14, pos=(x_positions[i], -0.17),
                           fillColor="#303030", lineColor="white", lineWidth=2)
        label = make_textbox2(win, option_text, (x_positions[i], -0.17), (button_width - 0.03, 0.12),
                              HI_FONT, 0.020, "white", "center", 0.002)
        stims.extend([rect, label])
        buttons.append(visual.Rect(win, width=button_width, height=0.14, pos=(x_positions[i], -0.17),
                                   fillColor=None, lineColor=None))
    return stims, buttons


def ask_button_question(writer, section, task_label, trial_num, subtrial_label,
                        title_en, title_hi, prompt_en, prompt_hi, option_labels,
                        marker_code, dass_item_num="", dass_subscale=""):
    clear_experiment_screen()
    send_marker(marker_code)
    onset_global = round(global_clock.getTime(), 6)
    onset_laptop = laptop_time_str()

    stims, buttons = make_button_page_stims(
        title_en,
        title_hi,
        prompt_en,
        prompt_hi,
        option_labels,
    )
    selected_value = None

    mouse.clickReset()
    event.clearEvents(eventType="keyboard")

    while selected_value is None:
        if QUIT_KEY in event.getKeys(keyList=[QUIT_KEY]):
            cleanup(win)

        for stim in stims:
            stim.draw()
        win.flip()

        if mouse.getPressed()[0]:
            for idx, button in enumerate(buttons):
                if button.contains(mouse):
                    selected_value = str(idx if section == "dass21" else idx + 1)
                    break
            while mouse.getPressed()[0]:
                core.wait(0.01)

    response_global = round(global_clock.getTime(), 6)
    response_laptop = laptop_time_str()
    write_row(writer, {
        "section": section,
        "trial_num": trial_num,
        "subtrial_num": subtrial_label,
        "phase": "question",
        "stim_onset_global": onset_global,
        "stim_onset_laptop_time": onset_laptop,
        "response_global": response_global,
        "response_key": selected_value,
        "response_laptop_time": response_laptop,
        "event_summary": "%s=%s | response=%s" % (task_label, subtrial_label, selected_value),
        "responded": 1,
        "marker_event_label": subtrial_label,
        "marker_code": marker_code,
        "survey_task": task_label,
        "survey_item": subtrial_label,
        "survey_response": selected_value,
        "task_order": task_order_label,
        "dass_item_num": dass_item_num,
        "dass_subscale": dass_subscale,
        "dass_response": selected_value if section == "dass21" else "",
    })
    return selected_value


def run_dass21(writer, trial_num):
    wait_for_bilingual_start(
        "Questionnaire\n\n"
        "Please read each statement and click the option that indicates how much it applied to you.\n\n"
        "Press SPACE to start.",
               "प्रश्नावली\n\n"
        "कृपया प्रत्येक कथन पढ़ें और वह आप पर कितना लागू हुआ, उसके अनुसार विकल्प चुनें।\n\n"
        "शुरू करने के लिए SPACE बटन दबाएँ।",
    )
    send_and_log_marker(writer, "dass21", trial_num, "section_start_dass21",
                        MARKERS["section_start_dass21"], summary="start_dass21")
    scores = {"depression": 0, "anxiety": 0, "stress": 0}
    for item_num, subscale, item_en, item_hi in DASS21_ITEMS:
        response = ask_button_question(
            writer, "dass21", "dass21", trial_num, "dass21_item_%02d" % item_num,
            "DASS-21 item %d of 21" % item_num,
            "DASS-21 प्रश्न %d / 21" % item_num,
            item_en,
            item_hi,
            DASS_OPTIONS,
            MARKERS["dass21_response"],
            item_num,
            subscale,
        )
        scores[subscale] += int(response)
    send_and_log_marker(writer, "dass21", trial_num, "section_end_dass21",
                        MARKERS["section_end_dass21"],
                        summary="depression=%d | anxiety=%d | stress=%d" % (
                            scores["depression"], scores["anxiety"], scores["stress"]))
    return trial_num + 1


def run_post_task_survey(writer, task_label, trial_num):
    send_and_log_marker(writer, "survey", trial_num, "survey_start",
                        MARKERS["survey_start"], summary="survey_start_for_%s" % task_label)
    ask_button_question(writer, "survey", task_label, trial_num, "difficulty",
                        "Post-task survey", "कार्य के बाद प्रश्नावली",
                        "How difficult was the last task?", "पिछला कार्य कितना कठिन था?",
                        ["1\nVery easy\nबहुत आसान", "2\nEasy\nआसान", "3\nModerate\nमध्यम",
                         "4\nDifficult\nकठिन", "5\nVery difficult\nबहुत कठिन"],
                        MARKERS["survey_difficulty"])
    ask_button_question(writer, "survey", task_label, trial_num, "stress_relax",
                        "Post-task survey", "कार्य के बाद प्रश्नावली",
                        "How calm or stressed did you feel during the last task?",
                        "पिछले कार्य के दौरान आप कितना शांत या तनावग्रस्त महसूस कर रहे थे?",
                        ["1\nVery relaxed\nबहुत शांत", "2\nRelaxed\nशांत", "3\nNeutral\nतटस्थ",
                         "4\nStressed\nतनावग्रस्त", "5\nVery stressed\nबहुत तनावग्रस्त"],
                        MARKERS["survey_stress_relax"])
    ask_button_question(writer, "survey", task_label, trial_num, "focus",
                        "Post-task survey", "कार्य के बाद प्रश्नावली",
                        "How focused were you during the last task?",
                        "पिछले कार्य के दौरान आपका ध्यान कितना केंद्रित था?",
                        ["1\nNot at all\nबिल्कुल नहीं", "2\nSlightly\nथोड़ा", "3\nModerately\nमध्यम रूप से",
                         "4\nHighly\nबहुत अधिक", "5\nExtremely\nअत्यधिक"],
                        MARKERS["survey_focus"])
    ask_button_question(writer, "survey", task_label, trial_num, "mental_effort",
                        "Post-task survey", "कार्य के बाद प्रश्नावली",
                        "How much mental effort did the last task require?",
                        "पिछले कार्य में आपको कितना मानसिक प्रयास करना पड़ा?",
                        ["1\nVery low\nबहुत कम", "2\nLow\nकम", "3\nModerate\nमध्यम",
                         "4\nHigh\nअधिक", "5\nVery high\nबहुत अधिक"],
                        MARKERS["survey_mental_effort"])
    ask_button_question(writer, "survey", task_label, trial_num, "fatigue",
                        "Post-task survey", "कार्य के बाद प्रश्नावली",
                        "How fatigued or sleepy did you feel during the last task?",
                        "पिछले कार्य के दौरान आप कितनी थकान या नींद महसूस कर रहे थे?",
                        ["1\nNot at all\nबिल्कुल नहीं", "2\nSlightly\nथोड़ी", "3\nModerately\nमध्यम",
                         "4\nVery\nबहुत", "5\nExtremely\nअत्यधिक"],
                        MARKERS["survey_fatigue"])
    send_and_log_marker(writer, "survey", trial_num, "survey_end",
                        MARKERS["survey_end"], summary="survey_end_for_%s" % task_label)
    wait_for_bilingual_start("Thank you.\n\nPress SPACE to continue.",
                             "धन्यवाद।\n\nआगे बढ़ने के लिए SPACE बटन दबाएँ।")
    return trial_num + 1


def run_baseline_rest(writer, trial_num, duration_s):
    wait_for_bilingual_start(
        "Relax phase\n\n"
        "This is the baseline recording.\n\n"
        "When you start, please close your eyes, remain still, and relax quietly.\n"
        "Breathe normally.\n\n"
        "The relax phase will last 5 minutes.\n"
        "A beep will play when it is over.\n\n"
        "Press SPACE to start the relax phase.",
        "विश्राम चरण\n\n"
        "यह बेसलाइन रिकॉर्डिंग है।\n\n"
        "शुरू होने पर कृपया अपनी आँखें बंद रखें, स्थिर बैठें और शांत रहें।\n"
        "सामान्य रूप से साँस लें।\n\n"
        "यह विश्राम चरण 5 मिनट तक चलेगा।\n"
        "समाप्त होने पर एक ध्वनि बजेगी।\n\n"
        "विश्राम चरण शुरू करने के लिए SPACE बटन दबाएँ।",
    )
    send_and_log_marker(writer, "baseline_rest", trial_num, "section_start_baseline",
                        MARKERS["section_start_baseline"], summary="eyes_closed_baseline_start")
    rest_start_global = round(global_clock.getTime(), 6)
    rest_start_laptop = laptop_time_str()
    show_tk_bilingual(
        "Please close your eyes and relax for 5 min.",
        "कृपया अपनी आँखें बंद करें और 5 मिनट तक आराम करें।",
        wait_for_space=False,
        duration_s=duration_s,
    )
    end_beep.play()
    core.wait(BEEP_DURATION + 0.10)
    rest_end_global = round(global_clock.getTime(), 6)
    rest_end_laptop = laptop_time_str()
    send_and_log_marker(writer, "baseline_rest", trial_num, "section_end_baseline",
                        MARKERS["section_end_baseline"], summary="eyes_closed_baseline_end")
    write_row(writer, {
        "section": "baseline_rest",
        "trial_num": trial_num,
        "phase": "rest",
        "stim_onset_global": rest_start_global,
        "stim_onset_laptop_time": rest_start_laptop,
        "event_summary": "baseline_rest_start=%s | baseline_rest_end=%s | duration=300s | eyes_closed=1" % (
            rest_start_laptop, rest_end_laptop),
        "section_end_global": rest_end_global,
        "section_end_laptop_time": rest_end_laptop,
        "task_order": task_order_label,
    })
    wait_for_bilingual_start(
        "The relax phase is complete.\n\nPlease open your eyes.\n\nPress SPACE to continue to the task.",
        "विश्राम चरण पूरा हो गया है।\n\nकृपया अपनी आँखें खोलें।\n\nकार्य जारी रखने के लिए SPACE बटन  दबाएँ।",
    )
    return trial_num + 1


exp_info = {
    "participant": "",
    "session": "001",
    "Age (in Years)": "",
    "Gender(M/F)": "",
    "Meditation Experience (Months)": "",
    "Are you vegetarian(1)/ non-vegetarian(2)eggetarian(3)": "Type 1 or 2 or 3",
    "Do you do any kind of intoxication": "alcohol, smoking",
    "Do you drink coffee": "",
    "Do you drink tea": "",
    "Hour of sleep (total in 24 hrs)": "",
    "Sleep Quality": "Best, Good, bad, Very bad",
}
dlg = gui.DlgFromDict(exp_info, title="Aging EEG/HRV Task")
if not dlg.OK:
    core.quit()

exp_info["date"] = data.getDateStr()
filename = "aging_task_%s_%s_%s.csv" % (exp_info["participant"], exp_info["session"], exp_info["date"])
csv_path = os.path.join(os.getcwd(), filename)

win = visual.Window(size=WINDOW_SIZE, fullscr=FULLSCREEN, color=BG_COLOR, units="height", waitBlanking=True)
mouse = event.Mouse(win=win)
fixation = visual.TextStim(win, text="+", color="white", height=0.06)
blue_circle = visual.Circle(win, radius=0.08, fillColor="blue", lineColor="blue")
red_triangle = visual.ShapeStim(win, vertices=[(-0.08, -0.06), (0.08, -0.06), (0.0, 0.10)],
                                fillColor="red", lineColor="red")
white_flash = visual.Rect(win, width=0.18, height=0.18, fillColor="white", lineColor="white")

standard_wave = make_tone_wave(STANDARD_FREQ, TONE_DURATION, TONE_SAMPLE_RATE, TONE_VOLUME, TONE_RAMP_SEC)
high_wave = make_tone_wave(HIGH_FREQ, TONE_DURATION, TONE_SAMPLE_RATE, TONE_VOLUME, TONE_RAMP_SEC)
beep_wave = make_tone_wave(BEEP_FREQ, BEEP_DURATION, TONE_SAMPLE_RATE, TONE_VOLUME, TONE_RAMP_SEC)
standard_tone = sound.Sound(value=standard_wave, sampleRate=TONE_SAMPLE_RATE, stereo=True, hamming=False)
high_tone = sound.Sound(value=high_wave, sampleRate=TONE_SAMPLE_RATE, stereo=True, hamming=False)
end_beep = sound.Sound(value=beep_wave, sampleRate=TONE_SAMPLE_RATE, stereo=True, hamming=False)
standard_tone.play()
core.wait(0.05)
standard_tone.stop(reset=True)

fieldnames = list(make_blank_row().keys())
global_clock = core.Clock()
oddball_phase1 = build_oddball_trials_from_counts(PHASE1_COUNTS)
oddball_phase2 = build_oddball_trials_from_counts(PHASE2_COUNTS)
oddball_phase3 = build_oddball_trials_from_counts(PHASE3_COUNTS)


def run_oddball_block(writer, trials, phase_name, start_trial_num, iti_low, iti_high):
    trial_counter = start_trial_num
    for trial in trials:
        shape_stim = red_triangle if trial["shape_name"] == "red_triangle" else blue_circle
        if trial["tone_name"] == "high_tone":
            tone_stim = high_tone
        elif trial["tone_name"] == "standard_tone":
            tone_stim = standard_tone
        else:
            tone_stim = None
        responded = 0
        response_key = ""
        response_laptop_time = ""
        response_global = ""
        rt = ""
        accuracy = 0
        response_outcome = ""
        draw_fixation_and_wait(win, fixation, random.uniform(iti_low, iti_high))
        event.clearEvents(eventType="keyboard")
        stop_all_tones()
        trial_clock = core.Clock()
        shape_stim.draw()
        win.callOnFlip(trial_clock.reset)
        win.callOnFlip(send_marker, trial["condition_code"])
        if tone_stim is not None:
            win.callOnFlip(schedule_tone_on_flip, win, tone_stim)
        win.flip()
        stim_onset_global = round(global_clock.getTime(), 6)
        stim_onset_laptop_time = laptop_time_str()
        while trial_clock.getTime() < ODD_STIM_DURATION:
            keys = event.getKeys(keyList=[RESPONSE_KEY, QUIT_KEY], timeStamped=trial_clock)
            for key, key_time in keys:
                if key == QUIT_KEY:
                    cleanup(win)
                if key == RESPONSE_KEY and not responded:
                    responded = 1
                    response_key = key
                    response_laptop_time = laptop_time_str()
                    response_global = round(global_clock.getTime(), 6)
                    rt = round(key_time, 4)
        fixation.draw()
        win.flip()
        while trial_clock.getTime() < ODD_TOTAL_TRIAL_DURATION:
            keys = event.getKeys(keyList=[RESPONSE_KEY, QUIT_KEY], timeStamped=trial_clock)
            for key, key_time in keys:
                if key == QUIT_KEY:
                    cleanup(win)
                if key == RESPONSE_KEY and not responded:
                    responded = 1
                    response_key = key
                    response_laptop_time = laptop_time_str()
                    response_global = round(global_clock.getTime(), 6)
                    rt = round(key_time, 4)
        if trial["any_target"] == 1 and responded == 1:
            accuracy = 1
            response_outcome = "hit"
        elif trial["any_target"] == 1 and responded == 0:
            accuracy = 0
            response_outcome = "miss"
        elif trial["any_target"] == 0 and responded == 1:
            accuracy = 0
            response_outcome = "false_alarm"
        else:
            accuracy = 1
            response_outcome = "correct_rejection"
        event_summary = "stim=%s | response=%s | rt=%s" % (
            stim_onset_laptop_time, response_laptop_time if responded else "none", rt if responded else "none")
        write_row(writer, {
            "section": "oddball",
            "trial_num": trial_counter,
            "phase": phase_name,
            "trial_type": trial["trial_type"],
            "condition_label": trial["condition_label"],
            "condition_code": trial["condition_code"],
            "visual_type": trial["visual_type"],
            "audio_type": trial["audio_type"],
            "audio_present": trial["audio_present"],
            "shape_name": trial["shape_name"],
            "tone_name": trial["tone_name"],
            "visual_target": trial["visual_target"],
            "audio_target": trial["audio_target"],
            "any_target": trial["any_target"],
            "both_targets": trial["both_targets"],
            "stim_onset_global": stim_onset_global,
            "stim_onset_laptop_time": stim_onset_laptop_time,
            "response_global": response_global,
            "response_key": response_key,
            "response_laptop_time": response_laptop_time,
            "event_summary": event_summary,
            "rt": rt,
            "responded": responded,
            "accuracy": accuracy,
            "response_outcome": response_outcome,
            "marker_code": trial["condition_code"],
            "task_order": task_order_label,
        })
        trial_counter += 1
    return trial_counter


def run_working_memory_block(writer, start_trial_num):
    trial_counter = start_trial_num
    for block_idx in range(WM_BLOCKS):
        seq_len = 1 if block_idx < 3 else 2 if block_idx < 9 else 3
        sequence = [str(random.randint(1, 9)) for _ in range(seq_len)]
        draw_fixation_and_wait(win, fixation, 0.7)
        for pos, item in enumerate(sequence, start=1):
            event.clearEvents(eventType="keyboard")
            send_marker(MARKERS["wm_encode"])
            draw_image_text(item, (0.0, 0.0), (0.30, 0.18), EN_FONT, 96, "white", "center")
            win.flip()
            item_time = laptop_time_str()
            item_global = round(global_clock.getTime(), 6)
            core.wait(WM_ENCODE_ITEM_DUR)
            fixation.draw()
            win.flip()
            core.wait(WM_BLANK_DUR)
            write_row(writer, {
                "section": "working_memory",
                "trial_num": trial_counter,
                "subtrial_num": pos,
                "phase": "encode",
                "stim_onset_global": item_global,
                "stim_onset_laptop_time": item_time,
                "event_summary": "stim=%s" % item_time,
                "wm_sequence": "".join(sequence),
                "wm_length": seq_len,
                "marker_code": MARKERS["wm_encode"],
                "task_order": task_order_label,
            })
        send_marker(MARKERS["wm_probe"])
        typed = []
        probe_onset_global = round(global_clock.getTime(), 6)
        probe_onset_laptop = laptop_time_str()
        recall_response_time = ""
        recall_response_global = ""
        while len(typed) < seq_len:
            shown = "".join(typed) if typed else "_" * seq_len
            draw_image_text("Type the sequence:\n\n%s" % shown, (0.0, 0.03), (1.20, 0.36),
                            EN_FONT, 46, "white", "center")
            win.flip()
            keys = event.getKeys()
            for key in keys:
                if key == QUIT_KEY:
                    cleanup(win)
                elif key == "backspace" and typed:
                    typed.pop()
                elif key in [str(x) for x in range(10)] and len(typed) < seq_len:
                    typed.append(key)
                    if len(typed) == seq_len:
                        recall_response_time = laptop_time_str()
                        recall_response_global = round(global_clock.getTime(), 6)
                        break
        response = "".join(typed)
        correct = int(response == "".join(sequence))
        send_marker(MARKERS["wm_response"])
        draw_image_text("Correct" if correct else "Incorrect", (0.0, 0.0), (0.50, 0.14),
                        EN_FONT, 54, "lightgreen" if correct else "salmon", "center")
        win.flip()
        core.wait(WM_FEEDBACK_DUR)
        write_row(writer, {
            "section": "working_memory",
            "trial_num": trial_counter,
            "subtrial_num": "recall",
            "phase": "recall",
            "stim_onset_global": probe_onset_global,
            "stim_onset_laptop_time": probe_onset_laptop,
            "response_global": recall_response_global,
            "response_key": response,
            "response_laptop_time": recall_response_time,
            "event_summary": "response=%s" % recall_response_time,
            "responded": int(len(response) > 0),
            "accuracy": correct,
            "response_outcome": "correct" if correct else "incorrect",
            "wm_sequence": "".join(sequence),
            "wm_length": seq_len,
            "wm_correct": correct,
            "marker_code": MARKERS["wm_response"],
            "task_order": task_order_label,
        })
        trial_counter += 1
    return trial_counter


def run_simple_rt_block(writer, start_trial_num):
    trial_counter = start_trial_num
    for _ in range(RT_TRIALS):
        delay = random.uniform(1.0, 2.5)
        draw_fixation_and_wait(win, fixation, delay)
        event.clearEvents(eventType="keyboard")
        rt_clock = core.Clock()
        white_flash.draw()
        win.callOnFlip(rt_clock.reset)
        win.callOnFlip(send_marker, MARKERS["rt_flash"])
        win.flip()
        stim_onset_global = round(global_clock.getTime(), 6)
        stim_onset_laptop_time = laptop_time_str()
        responded = 0
        response_key = ""
        response_laptop_time = ""
        response_global = ""
        rt = ""
        while rt_clock.getTime() < RT_FLASH_DUR:
            keys = event.getKeys(keyList=[RESPONSE_KEY, QUIT_KEY], timeStamped=rt_clock)
            for key, key_time in keys:
                if key == QUIT_KEY:
                    cleanup(win)
                if key == RESPONSE_KEY and not responded:
                    responded = 1
                    response_key = key
                    response_laptop_time = laptop_time_str()
                    response_global = round(global_clock.getTime(), 6)
                    rt = round(key_time, 4)
        fixation.draw()
        win.flip()
        while rt_clock.getTime() < RT_RESP_WINDOW:
            keys = event.getKeys(keyList=[RESPONSE_KEY, QUIT_KEY], timeStamped=rt_clock)
            for key, key_time in keys:
                if key == QUIT_KEY:
                    cleanup(win)
                if key == RESPONSE_KEY and not responded:
                    responded = 1
                    response_key = key
                    response_laptop_time = laptop_time_str()
                    response_global = round(global_clock.getTime(), 6)
                    rt = round(key_time, 4)
        event_summary = "stim=%s | response=%s | rt=%s" % (
            stim_onset_laptop_time, response_laptop_time if responded else "none", rt if responded else "none")
        write_row(writer, {
            "section": "simple_rt",
            "trial_num": trial_counter,
            "phase": "rt",
            "trial_type": "visual_flash",
            "visual_type": "flash",
            "audio_type": "none",
            "audio_present": 0,
            "shape_name": "white_flash",
            "visual_target": 1,
            "audio_target": 0,
            "any_target": 1,
            "both_targets": 0,
            "stim_onset_global": stim_onset_global,
            "stim_onset_laptop_time": stim_onset_laptop_time,
            "response_global": response_global,
            "response_key": response_key,
            "response_laptop_time": response_laptop_time,
            "event_summary": event_summary,
            "rt": rt,
            "responded": responded,
            "accuracy": int(responded == 1),
            "response_outcome": "hit" if responded else "miss",
            "rt_flash_delay": round(delay, 4),
            "marker_code": MARKERS["rt_flash"],
            "task_order": task_order_label,
        })
        trial_counter += 1
    return trial_counter


def run_oddball_task(writer, trial_num):
    wait_for_bilingual_start(
        "Oddball task\n\n"
        "Press SPACE whenever a rare event occurs:\n"
        "- red triangle\n"
        "- high tone\n\n"
        "Audio will occur only on some trials.\n"
        "Sometimes the sound and shape will occur together.\n\n"
        "Press SPACE to start.",
        "ऑडबॉल कार्य\n\n"
        "जब भी कोई दुर्लभ घटना हो, SPACE बटन  दबाएँ:\n"
        "- लाल त्रिकोण\n"
        "- ऊँची ध्वनि\n\n"
        "ध्वनि केवल कुछ ट्रायल्स में आएगी।\n"
        "कभी-कभी ध्वनि और आकृति साथ में आएँगे।\n\n"
        "शुरू करने के लिए SPACE बटन दबाएँ।",
    )
    send_and_log_marker(writer, "oddball", trial_num, "section_start_oddball",
                        MARKERS["section_start_oddball"], summary="start_main_oddball")
    trial_num = run_oddball_block(writer, oddball_phase1, "phase1_simple", trial_num, 0.45, 0.60)
    trial_num = run_oddball_block(writer, oddball_phase2, "phase2_mixed", trial_num, 0.35, 0.55)
    trial_num = run_oddball_block(writer, oddball_phase3, "phase3_complex", trial_num, 0.25, 0.45)
    send_and_log_marker(writer, "oddball", trial_num, "section_end_oddball",
                        MARKERS["section_end_oddball"], summary="end_main_oddball")
    return run_post_task_survey(writer, "oddball_main", trial_num)


def run_memory_task(writer, trial_num):
    wait_for_bilingual_start(
        "Memory task\n\n"
        "Watch the numbers carefully.\n"
        "After each sequence, type the numbers in the same order.\n"
        "Take as much time as you need.\n"
        "When you finish entering all digits, the task will move ahead automatically.\n\n"
        "Press SPACE to continue.",
        "स्मृति कार्य\n\n"
        "संख्याओं को ध्यान से देखें।\n"
        "हर श्रृंखला के बाद उन्हीं संख्याओं को उसी क्रम में टाइप करें।\n"
        "उत्तर देने के लिए आप जितना समय चाहें ले सकते हैं।\n"
        "सभी अंक दर्ज होने पर कार्य अपने आप आगे बढ़ जाएगा।\n\n"
        "आगे बढ़ने के लिए SPACE बटन  दबाएँ।",
    )
    send_and_log_marker(writer, "working_memory", trial_num, "section_start_wm",
                        MARKERS["section_start_wm"], summary="start_working_memory")
    trial_num = run_working_memory_block(writer, trial_num)
    send_and_log_marker(writer, "working_memory", trial_num, "section_end_wm",
                        MARKERS["section_end_wm"], summary="end_working_memory")
    return run_post_task_survey(writer, "working_memory", trial_num)


def run_reaction_task(writer, trial_num):
    wait_for_bilingual_start(
        "Reaction time task\n\n"
        "Press SPACE as quickly as possible whenever the white square flashes.\n\n"
        "Press SPACE to continue.",
        "प्रतिक्रिया समय कार्य\n\n"
        "जब भी सफेद वर्ग चमके, जितनी जल्दी हो सके SPACE बटन  दबाएँ।\n\n"
        "आगे बढ़ने के लिए SPACE बटन  दबाएँ।",
    )
    send_and_log_marker(writer, "simple_rt", trial_num, "section_start_rt",
                        MARKERS["section_start_rt"], summary="start_simple_rt")
    trial_num = run_simple_rt_block(writer, trial_num)
    send_and_log_marker(writer, "simple_rt", trial_num, "section_end_rt",
                        MARKERS["section_end_rt"], summary="end_simple_rt")
    return run_post_task_survey(writer, "simple_rt", trial_num)


TASKS = [
    ("oddball", run_oddball_task),
    ("working_memory", run_memory_task),
    ("simple_rt", run_reaction_task),
]
random.shuffle(TASKS)
task_order_label = ",".join([task_name for task_name, _ in TASKS])

wait_for_bilingual_start(
    "This task has several short parts.\n\n"
    "First, there will be a 5-minute relax phase.\n"
    "After that, the three tasks will appear in random order.\n\n"
    "Task parts:\n"
    "- Oddball task: press SPACE whenever you see a red triangle or hear a high tone.\n"
    "- Memory task: remember short number sequences.\n"
    "- Reaction task: press SPACE when you see a white flash.\n\n"
    "After each task, you will answer a short survey.\n\n"
    "At the end, you will complete the questionnaire.\n\n"
    "Press SPACE to begin.",
    "इस कार्य में कई छोटे भाग हैं।\n\n"
    "सबसे पहले 5 मिनट का विश्राम चरण होगा।\n"
    "उसके बाद तीन कार्य यादृच्छिक क्रम में आएँगे।\n\n"
    "कार्य के भाग:\n"
    "- ऑडबॉल कार्य: लाल त्रिकोण दिखाई देने या ऊँची ध्वनि सुनाई देने पर SPACE बटन दबाएँ।\n"
    "- स्मृति कार्य: छोटी संख्या-श्रृंखलाएँ याद रखें।\n"
    "- प्रतिक्रिया कार्य: सफेद वर्ग चमकने पर SPACE बटन दबाएँ।\n\n"
    "हर कार्य के बाद आपको एक छोटी प्रश्नावली भरनी होगी।\n\n"
    "अंत में आप  प्रश्नावली पूरी करेंगे।\n\n"
    "शुरू करने के लिए SPACE बटन दबाएँ।",
)

trial_num = 1
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    write_row(writer, {
        "section": "task_order",
        "trial_num": trial_num,
        "phase": "randomized_order",
        "event_summary": "task_order=%s" % task_order_label,
        "task_order": task_order_label,
    })
    trial_num = run_baseline_rest(writer, trial_num, BASELINE_REST_DUR)
    for task_name, task_func in TASKS:
        trial_num = task_func(writer, trial_num)
    trial_num = run_dass21(writer, trial_num)

show_tk_bilingual(
    "Task complete.\n\nThank you.",
    "कार्य पूरा हो गया।\n\nधन्यवाद।",
    wait_for_space=False,
    duration_s=3.0,
)
cleanup(win)
