import os
path = 'output/car_right/'

# car, boat, motorbike, bottle done

# cube, sphere, cylinder, cone, torus, suzanne

print path

numIDs = 500

lowers = []
face = 0
while face < numIDs:
    if face % 5000 == 0:
        print 'Checking: ', face
    if not (os.path.exists( path + str(face) + '_normals.png' ) and \
        os.path.exists( path + str(face) + '_mask.png' ) and \
        os.path.exists( path + str(face) + '_specular.png' )):
        print '\nMissing: ', face, '\n'
        lowers.append(face)
        face += 50
    else:
        face += 1





# lowers = [18700, 23000, 38000]

def makeDivisions(lowers):
    divisions = []
    for ind in range(len(lowers)):
        temp = [(i,i+10) for i in range(lowers[ind],lowers[ind]+50,10)]
        divisions.extend(temp)
    return divisions

divisions = makeDivisions(lowers)

print divisions