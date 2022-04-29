import pandas as pd

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

# On 6 Dec 2016, a commit was pushed which changed the behavior of Android
# Radio question answers. The commit was called "Gets nullable Integer
# answers from sliders and radio button questions"
# and can be found at
# https://github.com/onnela-lab/beiwe-android/commit/6341eb5498ceeffcb64d65c2dd2bcfdab9b982f2
ANDROID_NULLABLE_ANSWER_CHANGE_DATE = pd.to_datetime("2016-12-06")
