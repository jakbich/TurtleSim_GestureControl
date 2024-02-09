from setuptools import setup

package_name = 'gesture_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jakob',
    maintainer_email='j.d.bichler@student.tudelft.nl',
    description='Package to control TurtleSim using hand gestures',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gesture_publisher = gesture_control.gesture_publisher:main',
            'depth_publisher = gesture_control.depth_publisher:main'    
            ],
    },
)
