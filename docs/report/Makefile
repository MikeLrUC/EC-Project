REPORT_NAME=report
FOLDER=bin
.PHONY: default
default:
	@mkdir -p ${FOLDER}
	@pdflatex -output-directory=${FOLDER} ${REPORT_NAME}.tex
	@mv ${FOLDER}/${REPORT_NAME}.pdf ../