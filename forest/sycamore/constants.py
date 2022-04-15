# We want our default date to be farther in the past than any Beiwe data could
# have been collected, so we never cut off data by default. But, if we set our
# default date too far in the past, we would generate too many weekly survey
# timings
EARLIEST_DATE = "2010-01-01"

# load events & question types dictionary
QUESTION_TYPES_LOOKUP = {
    "Android": {"Checkbox Question": "checkbox",
                "Info Text Box": "info_text_box",
                "Open Response Question": "free_response",
                "Radio Button Question": "radio_button",
                "Slider Question": "slider"},
    "iOS": {"checkbox": "checkbox",
            "free_response": "free_response",
            "info_text_box": "info_text_box",
            "radio_button": "radio_button",
            "slider": "slider"}
}
