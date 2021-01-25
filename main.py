import face_extractor
import Codebook as cb
import Network as ntw

# extractor = face_extractor.FaceExtractor('faces', 'cropped_images')
# extractor.preprocess_images()

codebook = cb.Codebook()
codebook.create_codebook('cropped_images', 'Codebook_cell8x8')
codebook.Load_codebook_to_mem('Codebook_cell8x8')
# v_angle, h_angle = codebook.Estimate_angles_for_img('cropped_images/Person15/0person15115-30-75.jpg')
# print("Estimated orientation: Vertical= {}, Horizontal= {}".format(v_angle, h_angle))

dataset = ntw.prepare_data('Codebook_cell8x8')
ntw.create_model(dataset)

