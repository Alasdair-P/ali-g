#!bin/bash                                                                                                                        

#SBATCH --partition=small                                                                                                          

# set the number of nodes                                                                                                          
#SBATCH --nodes=1                                                                                                                  

# set max wallclock time                                                                                                           
#SBATCH --time=20:00:00                                                                                                            

# set name of job                                                                                                                  
#SBATCH --job-name=cifar-no-aug                                                                                                    

# set number of GPUs                                                                                                               
#SBATCH --gres=gpu:1                                                                                                              

# job array                                                                                                                        
#SBATCH --array=0-7                                                                                                                

# run the application                                                                                                              
./launch.sh
