import os, shutil
import hazelbean as hb
student_assignment_dir = '../../../Student Assignment Submissions/assignment_4'

for input_file in os.listdir(student_assignment_dir):
    if input_file.endswith('.py'):
        student_dir = os.path.join(student_assignment_dir, os.path.splitext(input_file)[0])
        try:
            os.mkdir(student_dir)
        except:
            pass
        input_path = os.path.join(student_assignment_dir, input_file)
        student_path = os.path.join(student_dir, input_file)
        # hb.copy_shutil_flex(input_path, student_path)
        print(input_path)
        os.system('python \"' + str(student_path) + '\"')


