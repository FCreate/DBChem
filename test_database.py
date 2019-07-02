def test_base(cursor):
    count = 0
    for row in cursor.execute('SELECT * FROM molecules'):
        if(len(row)!=0):
            print("Molecules DB stored")
            count += 1
        break

    for row in cursor.execute('SELECT * FROM tasks'):
        if(len(row)!=0):
            print("Tasks DB stored")
            count += 1
        break

    for row in cursor.execute('SELECT * FROM tasks_running'):
        if(len(row)!=0):
            print("Tasks running stored")
            count += 1
        break

    for row in cursor.execute('SELECT * FROM descriptors'):
        if (len(row)!=0):
            print("Descriptors stored")
            count += 1
        break

    for row in cursor.execute('SELECT * FROM descriptors_values'):
        if (len(row)!=0):
            print("Descriptors values stored")
            count += 1
        break

    for row in cursor.execute('SELECT * FROM endpoints'):
        if(len(row)!=0):
            print("Endpoints stored")
            count += 1
        break

    for row in cursor.execute('SELECT * FROM experimental_data'):
        if (len(row)!=0):
            print("Experimental data stored")
            count += 1
        break
    if(count == 7):
        print("Tests are passed")
    else:
        print("Tests not passed")