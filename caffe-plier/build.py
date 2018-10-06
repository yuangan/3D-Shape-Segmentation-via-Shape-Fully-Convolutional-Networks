def build():
	train_file=open("train.prototxt","r")
	solver_file=open("solver.prototxt","r")
	train=train_file.read()
	solver=solver_file.read()

	all_file_number=7

	for i in range(all_file_number):
    		new_train_file=open("./solve/train_"+str(i)+".prototxt","w")
    		new_solver_file=open("./solve/solver_"+str(i)+".prototxt","w")
   		temp_t=train.replace("{i}",str(i))
   		temp_s=solver.replace("{i}",str(i))
   		new_train_file.write(temp_t)
   		new_solver_file.write(temp_s)
    		new_train_file.close()
    		new_solver_file.close()

	train_file.close()
	solver_file.close()
