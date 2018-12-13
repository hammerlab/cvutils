from setuptools import setup

if __name__ == '__main__':
    setup(
        name='cvutils',
        version='0.0.1',
        description="Hammer Lab Computer Vision Utilities",
        author="Eric Czech",
        author_email="eric@hammerlab.org",
        url="",
        license="http://www.apache.org/licenses/LICENSE-2.0.html",
        classifiers=[
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python'
        ],
        install_requires=requires,
        packages=[
            'cvutils',
            'cvutils.augmentation',
            'cvutils.imagej',
            'cvutils.keras',
            'cvutils.mrcnn',
            'cvutils.rectlabel'
        ],
        package_data={},
        include_package_data=False
    )
