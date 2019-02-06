NVCC = nvcc

helloCUDA: kernel.cu host.cu  
	$(NVCC) -I ./ kernel.cu host.cu -o helloCUDA

      
run:
	./helloCUDA

clean:
	rm -rf helloCUDA
