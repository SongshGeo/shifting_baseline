# Changelog

All notable changes to Shifting Baseline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive API documentation with MkDocs
- Material theme for documentation
- Detailed examples for all modules
- Development and testing guides

### Changed
- Improved code organization and structure
- Enhanced error handling and validation
- Updated documentation format

### Fixed
- Various bug fixes and improvements

## [0.1.0] - 2025-01-08

### Added
- Initial release of Shifting Baseline library
- Core data processing modules
- Historical records analysis
- Climate reconstruction data handling
- Statistical comparison tools
- Agent-based modeling framework
- Calibration and mismatch analysis
- Comprehensive filtering and classification
- Utility functions and helpers

### Features
- **Data Module**: Loading and processing of historical and climate data
- **Comparison Module**: Advanced correlation analysis and statistical comparison
- **Calibration Module**: Mismatch analysis and validation tools
- **Filters Module**: Data filtering and classification functions
- **ABM Module**: Agent-based modeling for climate event recording
- **Process Module**: Data processing pipelines and workflows
- **Utils Module**: Utility functions and helper classes

### Dependencies
- Python 3.11+
- xarray >= 2023
- pandas >= 1.5
- numpy >= 1.20
- matplotlib >= 3.9.2
- scipy >= 1.9
- scikit-learn >= 1.0
- geopandas
- cartopy >= 0.24.1
- hydra-core ~ 1.3
- abses >= 0.7.5

### Documentation
- Comprehensive API documentation
- Usage examples and tutorials
- Development and contributing guides
- Testing documentation

## [0.0.1] - 2024-12-01

### Added
- Initial project setup
- Basic project structure
- Core dependencies
- Initial configuration files

### Features
- Project scaffolding
- Basic configuration management
- Initial documentation structure

---

## Version History

### Version 0.1.0 (2025-01-08)
- **Major Features**:
  - Complete data processing pipeline
  - Historical records analysis
  - Climate reconstruction data handling
  - Statistical comparison tools
  - Agent-based modeling framework
  - Calibration and mismatch analysis
  - Comprehensive filtering and classification

- **API Stability**: Initial stable API
- **Documentation**: Complete API documentation
- **Testing**: Comprehensive test suite
- **Performance**: Optimized for large datasets

### Version 0.0.1 (2024-12-01)
- **Initial Release**: Project setup and basic structure
- **Configuration**: Basic configuration management
- **Documentation**: Initial documentation framework

## Future Releases

### Planned Features
- Enhanced visualization capabilities
- Additional statistical methods
- Improved performance optimizations
- Extended data format support
- Advanced machine learning integration
- Real-time data processing
- Cloud deployment support

### Roadmap
- **v0.2.0**: Enhanced visualization and plotting
- **v0.3.0**: Advanced machine learning integration
- **v0.4.0**: Performance optimizations
- **v0.5.0**: Extended data format support
- **v1.0.0**: Stable API and production ready

## Migration Guide

### Upgrading from 0.0.1 to 0.1.0

#### Breaking Changes
- Complete API redesign
- New module structure
- Updated function signatures
- Changed configuration format

#### Migration Steps
1. Update import statements
2. Update function calls
3. Update configuration files
4. Test with new API

#### Example Migration

**Before (v0.0.1):**
```python
# Old API (example)
from shifting_baseline.old_module import old_function
result = old_function(data)
```

**After (v0.1.0):**
```python
# New API
from shifting_baseline.filters import classify_series
result = classify_series(data)
```

## Deprecation Notices

### Deprecated in 0.1.0
- No deprecated features in initial release

### Will be deprecated in future versions
- Specific features will be announced in future releases

## Security

### Security Updates
- Regular security updates for dependencies
- Vulnerability scanning
- Secure coding practices

### Reporting Security Issues
- Email: songshgeo@gmail.com
- GitHub Issues: For public security issues
- Private disclosure: For sensitive security issues

## Performance

### Performance Improvements
- Optimized data loading
- Memory-efficient processing
- Parallel processing support
- Caching mechanisms

### Benchmarking
- Performance benchmarks included
- Regular performance testing
- Optimization tracking

## Dependencies

### Core Dependencies
- Python 3.11+
- xarray >= 2023
- pandas >= 1.5
- numpy >= 1.20
- matplotlib >= 3.9.2
- scipy >= 1.9

### Optional Dependencies
- scikit-learn >= 1.0 (for ML features)
- geopandas (for geospatial analysis)
- cartopy >= 0.24.1 (for cartographic projections)
- abses >= 0.7.5 (for agent-based modeling)

### Development Dependencies
- pytest >= 8.3.3
- pytest-cov
- black
- flake8
- mypy
- pre-commit

## Contributing

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Contribution Guidelines
- Follow code style guidelines
- Write comprehensive tests
- Update documentation
- Follow semantic versioning

## Support

### Getting Help
- Documentation: [docs/](docs/)
- Examples: [examples/](examples/)
- Issues: [GitHub Issues](https://github.com/SongshGeo/shifting_baseline/issues)
- Email: songshgeo@gmail.com

### Community
- GitHub Discussions
- Email list
- Regular updates

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

### Contributors
- SongshGeo (Maintainer)
- Community contributors

### Dependencies
- Thanks to all open-source projects that make Shifting Baseline possible

### Research
- Based on climate reconstruction research
- Historical data analysis methods
- Agent-based modeling techniques

## Contact

- **Maintainer**: SongshGeo
- **Email**: songshgeo@gmail.com
- **GitHub**: https://github.com/SongshGeo
- **Website**: https://cv.songshgeo.com/

---

*This changelog is automatically updated with each release.*
