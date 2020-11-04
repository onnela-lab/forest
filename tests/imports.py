

def test_imports(package_name, subpackage_names):
    n_success = 0
    n_failure = 0
    print('Testing imports from %s...' % package_name)
    for spn in subpackage_names:
        test_import = 'from %s.%s import *' % (package_name, spn)
        try:
            exec(test_import)
            n_success += 1
        except:
            print('  Failed to import %s.%' % (package_name, spn))
            n_failure += 1        
    print('Successful imports: %s' % n_success)
    print('Failed imports: %s \n' % n_failure)



'''
Test imports for forest.poplar
'''





'''
Test imports for forest.poplar
'''
test_imports('forest.poplar', 
             ['classes', 'constants', 'functions', 'legacy', 'raw'])
