OUT_DIR = out
GEN_DIR = gen

$(OUT_DIR)/relazione.pdf: $(wildcard *.md) | $(OUT_DIR)
	pandoc -t latex --pdf-engine=xelatex --toc --listings -o $@ $(sort $^)

$(OUT_DIR) $(GEN_DIR):
	mkdir -p $@

clean:
	rm -rf $(OUT_DIR) $(GEN_DIR)
	
