library(ActCR)

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
stopifnot(length(script_arg) == 1)
validation_dir <- dirname(normalizePath(
    sub("^--file=", "", script_arg)
))

files <- sort(list.files(
    file.path(validation_dir, "profiles"),
    pattern = "_1min\\.csv$",
    full.names = TRUE
))

results <- list()

for (file in files) {
    name <- sub("_1min\\.csv$", "", basename(file))
    activity <- read.csv(file)$activity

    stopifnot(length(activity) == 7 * 1440)

    days <- matrix(activity, nrow = 7, byrow = TRUE)
    profile <- colMeans(days)

    hourly <- rowMeans(
        matrix(activity, ncol = 60, byrow = TRUE)
    )

    for (resolution in c("1min", "1h")) {
        if (resolution == "1min") {
            values <- activity
            day_table <- as.data.frame(days)
            window <- 1
        } else {
            values <- hourly
            day_table <- as.data.frame(
                matrix(hourly, nrow = 7, byrow = TRUE)
            )
            window <- 60
        }

        windows <- RA(
            profile,
            window = window,
            method = "average"
        )

        metrics <- c(
            IS = IS(day_table),
            IV = IV(values),
            L5 = windows$L5,
            M10 = windows$M10,
            RA = windows$RA
        )

        results[[length(results) + 1]] <- data.frame(
            source = "ActCR-0.4.0",
            profile = name,
            resolution = resolution,
            metric = names(metrics),
            value = as.numeric(metrics)
        )

        cat("Completed:", name, resolution, "\n")
    }
}

reference <- do.call(rbind, results)

write.csv(
    reference,
    file.path(validation_dir, "reference_actcr.csv"),
    row.names = FALSE
)

cat("\nTotal reference results:", nrow(reference), "\n")
print(head(reference, 10))
